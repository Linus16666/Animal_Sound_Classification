/*
 * esp32s3.ino — ESP32-S3 standalone animal-sound classifier
 *
 * Hardware
 * --------
 * Board : ESP32S3 Dev Module (N16R8: 16 MB flash, 8 MB OPI PSRAM)
 * UART0 : USB/Serial — debug output (GPIO43/44)
 * UART1 : GPIO16(RX1) ← Arduino TX,  GPIO17(TX1) → Arduino RX
 *
 * Wiring (Arduino Uno ↔ level-shifter ↔ ESP32-S3)
 * ------------------------------------------------
 * Arduino Pin 1 (TX, 5 V) → 5V→3.3V divider (10kΩ+20kΩ) → GPIO16
 * Arduino Pin 0 (RX, 5 V) ← GPIO17  (3.3 V HIGH accepted by Uno RX)
 * GND ←──────────────────────────────────────────────────────────── GND
 *
 * Arduino IDE settings
 * --------------------
 *  Board      : "ESP32S3 Dev Module"
 *  PSRAM      : "OPI PSRAM"
 *  Flash Size : "16MB (128Mb)"
 *  Partition  : custom partitions.csv in sketch directory (app0=4MB, model=7.5MB raw)
 *  USB CDC On Boot: Enabled
 *  Upload Speed: 921600
 *
 * After flashing firmware, flash the model to the raw partition:
 *   esptool.py --chip esp32s3 -p /dev/ttyACM1 -b 921600 \
 *              write_flash 0x410000 inference_mic/crnn.tflite
 *
 * Required libraries (Arduino Library Manager)
 * ---------------------------------------------
 *  esp-tflite-micro  (by Espressif Systems)
 *
 * Serial protocol with Arduino
 * ----------------------------
 *  ESP32-S3 → Arduino:  "START\n"   start ADC streaming
 *  Arduino  → ESP32-S3: <int>\r\n   raw ADC values 0–1023, ~2200/s
 *  ESP32-S3 → Arduino:  "STOP\n"    stop ADC streaming
 *
 * WiFi AP
 * -------
 *  SSID: AnimalClassifier   Password: soundsound
 *  http://192.168.4.1/        — web page (auto-refreshes every 3 s)
 *  http://192.168.4.1/result  — JSON {"label":"dog","confidence":87.3}
 */

/* ── TFLite Micro ────────────────────────────────────────────────────── */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── Flash partition mmap ────────────────────────────────────────────── */
#include "esp_partition.h"

/* ── WiFi / WebServer ────────────────────────────────────────────────── */
#include <WiFi.h>
#include <WebServer.h>

/* ── Project headers ─────────────────────────────────────────────────── */
#include "mel_spectrogram.h"   /* compute_mel_spectrogram(), dc_remove_normalize() */
#include "labels.h"            /* ESC50_LABELS[]                                   */

/* ── Configuration ───────────────────────────────────────────────────── */
#define CAPTURE_MS      5000        /* recording window in ms              */
#define BAUD_RATE       115200
#define UART1_RX_PIN    16
#define UART1_TX_PIN    17
#define ARENA_SIZE      (6 * 1024 * 1024)   /* 6 MB TFLite tensor arena   */
#define RESAMPLE_N      220500      /* 5 s × 44100 Hz target samples       */

/* Maximum raw samples from Arduino in 5 s at ~2200 samples/s + margin  */
#define MAX_RAW_SAMPLES 15000

/* ── WiFi credentials ────────────────────────────────────────────────── */
static const char * WIFI_SSID = "AnimalClassifier";
static const char * WIFI_PASS = "soundsound";

/* ── Global TFLite objects ───────────────────────────────────────────── */
namespace {
    const tflite::Model         * model_ptr      = nullptr;
    tflite::MicroInterpreter    * interpreter     = nullptr;
    TfLiteTensor                * input_tensor    = nullptr;
    uint8_t                     * tensor_arena    = nullptr;
}   /* namespace */

/* ── PSRAM audio buffers ─────────────────────────────────────────────── */
static int16_t  * raw_samples    = nullptr;   /* ADC values 0–1023         */
static float    * resampled      = nullptr;   /* float32[RESAMPLE_N]       */
/* mel spectrogram: [MEL_N_MELS_DEF][MEL_T_FRAMES] stored row-major       */
static float    (*mel_db)[MEL_T_FRAMES] = nullptr;

/* ── Serial line buffer ──────────────────────────────────────────────── */
static char     line_buf[32];
static int      line_len = 0;

/* ── Latest inference result ─────────────────────────────────────────── */
static char  last_label[64]  = "Waiting...";
static float last_confidence = 0.0f;

/* ── Web server ──────────────────────────────────────────────────────── */
WebServer server(80);

/* ── Web handlers ────────────────────────────────────────────────────── */
static void handle_root()
{
    char page[1024];
    snprintf(page, sizeof(page),
        "<!DOCTYPE html>"
        "<html><head>"
        "<meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='3'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Animal Classifier</title>"
        "<style>"
        "body{background:#1a1a2e;color:#eee;font-family:sans-serif;"
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;min-height:100vh;margin:0}"
        "h1{font-size:1.4rem;color:#aaa;margin-bottom:2rem}"
        ".label{font-size:3rem;font-weight:bold;color:#e94560;margin-bottom:1rem}"
        ".conf{font-size:1.8rem;color:#0f9b8e}"
        "</style>"
        "</head><body>"
        "<h1>Animal Sound Classifier</h1>"
        "<div class='label'>%s</div>"
        "<div class='conf'>%.1f%%</div>"
        "</body></html>",
        last_label, last_confidence);
    server.send(200, "text/html", page);
}

static void handle_result()
{
    char json[128];
    snprintf(json, sizeof(json),
             "{\"label\":\"%s\",\"confidence\":%.1f}",
             last_label, last_confidence);
    server.send(200, "application/json", json);
}

/* ════════════════════════════════════════════════════════════════════════
   setup()
   ════════════════════════════════════════════════════════════════════════ */
void setup()
{
         pinMode(2, OUTPUT);
      for (int i = 0; i < 5; i++) { digitalWrite(2, HIGH); delay(200); digitalWrite(2, LOW); delay(200); }
 
    Serial.begin(BAUD_RATE);   /* UART0 → USB debug */
    { uint32_t t = millis(); while (!Serial && millis() - t < 5000) delay(10); }
    Serial.println("\n\n== ESP32-S3 Animal Sound Classifier ==");

    /* UART1 → Arduino */
    Serial1.begin(BAUD_RATE, SERIAL_8N1, UART1_RX_PIN, UART1_TX_PIN);

    /* ── WiFi AP ──────────────────────────────────────────────────────── */
    WiFi.softAP(WIFI_SSID, WIFI_PASS);
    Serial.printf("WiFi AP started. SSID: %s  IP: %s\n",
                  WIFI_SSID, WiFi.softAPIP().toString().c_str());

    server.on("/",       handle_root);
    server.on("/result", handle_result);
    server.begin();
    Serial.println("HTTP server started.");

    /* ── Allocate PSRAM buffers ──────────────────────────────────────── */
    tensor_arena = (uint8_t *)ps_malloc(ARENA_SIZE);
    raw_samples  = (int16_t *)ps_malloc(MAX_RAW_SAMPLES * sizeof(int16_t));
    resampled    = (float   *)ps_malloc(RESAMPLE_N       * sizeof(float));
    mel_db       = (float (*)[MEL_T_FRAMES])
                   ps_malloc(MEL_N_MELS_DEF * MEL_T_FRAMES * sizeof(float));

    if (!tensor_arena || !raw_samples || !resampled || !mel_db) {
        Serial.println("FATAL: PSRAM allocation failed. Check PSRAM mode = OPI PSRAM.");
        while (true) { delay(1000); }
    }

    Serial.printf("PSRAM free after alloc: %u bytes\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    /* ── Load TFLite model from raw flash partition ───────────────────── */
    tflite::InitializeTarget();

    const esp_partition_t * mp = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!mp) {
        Serial.println("FATAL: 'model' partition not found. Check partitions.csv and flash.");
        while (true) { delay(1000); }
    }

    const void                * model_mmap_ptr = nullptr;
    esp_partition_mmap_handle_t mmap_handle;
    esp_err_t err = esp_partition_mmap(mp, 0, mp->size,
                                       ESP_PARTITION_MMAP_DATA,
                                       &model_mmap_ptr, &mmap_handle);
    if (err != ESP_OK) {
        Serial.printf("FATAL: esp_partition_mmap() failed: %d\n", err);
        while (true) { delay(1000); }
    }

    model_ptr = tflite::GetModel(model_mmap_ptr);
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("FATAL: Model schema version %d != expected %d\n",
                      model_ptr->version(), TFLITE_SCHEMA_VERSION);
        while (true) { delay(1000); }
    }

    /* ── Op resolver ─────────────────────────────────────────────────── */
    static tflite::MicroMutableOpResolver<25> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddTranspose();
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddBatchMatMul();
    resolver.AddStridedSlice();
    resolver.AddSlice();
    resolver.AddRelu();
    resolver.AddConcatenation();
    resolver.AddSplit();
    resolver.AddSplitV();
    resolver.AddLogistic();
    resolver.AddTanh();
    resolver.AddPack();
    resolver.AddUnpack();
    resolver.AddReverseV2();
    resolver.AddSum();

    /* ── Build interpreter ───────────────────────────────────────────── */
    static tflite::MicroInterpreter static_interpreter(
        model_ptr, resolver, tensor_arena, ARENA_SIZE);
    interpreter = &static_interpreter;

    TfLiteStatus alloc_status = interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("FATAL: AllocateTensors() failed — try increasing ARENA_SIZE.");
        while (true) { delay(1000); }
    }

    input_tensor = interpreter->input(0);
    Serial.printf("TFLite model loaded. Arena used: %u bytes\n",
                  (unsigned)interpreter->arena_used_bytes());
    Serial.printf("Input  shape: [%d, %d, %d, %d]\n",
                  input_tensor->dims->data[0], input_tensor->dims->data[1],
                  input_tensor->dims->data[2], input_tensor->dims->data[3]);

    Serial.println("Ready. Starting first capture in 1 s...");
    delay(1000);
}

/* ════════════════════════════════════════════════════════════════════════
   Linear interpolation resample: src[src_n] → dst[dst_n]
   ════════════════════════════════════════════════════════════════════════ */
static void resample_linear(const float * src, int src_n,
                             float       * dst, int dst_n)
{
    float ratio = (float)(src_n - 1) / (float)(dst_n - 1);
    for (int i = 0; i < dst_n; i++) {
        float pos  = i * ratio;
        int   lo   = (int)pos;
        float frac = pos - lo;
        int   hi   = (lo + 1 < src_n) ? lo + 1 : lo;
        dst[i] = src[lo] + frac * (src[hi] - src[lo]);
    }
}

/* ════════════════════════════════════════════════════════════════════════
   Read one '\n'-terminated line from Serial1 into line_buf.
   Returns true if a complete line was received.
   ════════════════════════════════════════════════════════════════════════ */
static bool read_line_nonblock()
{
    while (Serial1.available()) {
        char c = (char)Serial1.read();
        if (c == '\n') {
            line_buf[line_len] = '\0';
            line_len = 0;
            return true;
        }
        if (c != '\r' && line_len < (int)(sizeof(line_buf) - 1)) {
            line_buf[line_len++] = c;
        }
    }
    return false;
}

/* ════════════════════════════════════════════════════════════════════════
   loop()
   ════════════════════════════════════════════════════════════════════════ */
void loop()
{
    /* ── 1. Tell Arduino to start sending ADC data ────────────────────── */
    Serial1.print("START\n");
    Serial1.flush();
    Serial.println("Capturing 5 s...");

    /* ── 2. Collect ADC samples for CAPTURE_MS milliseconds ──────────── */
    int           n_raw = 0;
    unsigned long t0    = millis();

    while ((millis() - t0) < CAPTURE_MS) {
        server.handleClient();   /* keep web server responsive during capture */
        if (read_line_nonblock()) {
            char * end;
            long   val = strtol(line_buf, &end, 10);
            if (end != line_buf && val >= 0 && val <= 1023
                && n_raw < MAX_RAW_SAMPLES) {
                raw_samples[n_raw++] = (int16_t)val;
            }
        }
    }

    /* ── 3. Stop ADC stream ───────────────────────────────────────────── */
    Serial1.print("STOP\n");
    Serial1.flush();

    float actual_rate = (float)n_raw / (CAPTURE_MS / 1000.0f);
    Serial.printf("Captured %d samples at %.0f Hz\n", n_raw, actual_rate);

    if (n_raw < 100) {
        Serial.println("Not enough samples — retrying in 1 s.");
        delay(1000);
        server.handleClient();
        return;
    }

    /* ── 4. Convert int16 ADC → float, DC-remove, normalize ─────────── */
    {
        static float raw_float[MAX_RAW_SAMPLES];
        for (int i = 0; i < n_raw; i++)
            raw_float[i] = (float)raw_samples[i];
        dc_remove_normalize(raw_float, n_raw);

        /* ── 5. Resample to 44100 Hz ───────────────────────────────── */
        resample_linear(raw_float, n_raw, resampled, RESAMPLE_N);
    }

    /* ── 6. Compute mel spectrogram ──────────────────────────────────── */
    Serial.println("Computing mel spectrogram...");
    int rc = compute_mel_spectrogram(resampled, RESAMPLE_N, mel_db);
    if (rc != 0) {
        Serial.println("ERR: mel spectrogram alloc failed.");
        delay(1000);
        server.handleClient();
        return;
    }

    /* ── 7. Fill input tensor [1, 1, 64, 431] ────────────────────────── */
    float * inp = input_tensor->data.f;
    for (int m = 0; m < MEL_N_MELS_DEF; m++)
        for (int t = 0; t < MEL_T_FRAMES; t++)
            inp[m * MEL_T_FRAMES + t] = mel_db[m][t];

    /* ── 8. Run inference ────────────────────────────────────────────── */
    Serial.println("Running inference...");
    unsigned long ti = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("ERR: Invoke() failed.");
        delay(1000);
        server.handleClient();
        return;
    }
    Serial.printf("Inference: %lu ms\n", millis() - ti);

    /* ── 9. Softmax output → argmax + confidence ─────────────────────── */
    TfLiteTensor * output_tensor = interpreter->output(0);
    float        * out           = output_tensor->data.f;
    int            n_classes     = output_tensor->dims->data[1];

    float max_logit = out[0];
    for (int i = 1; i < n_classes; i++)
        if (out[i] > max_logit) max_logit = out[i];

    float sum_exp = 0.0f;
    static float probs[50];
    for (int i = 0; i < n_classes; i++) {
        probs[i] = expf(out[i] - max_logit);
        sum_exp  += probs[i];
    }

    int   class_idx  = 0;
    float confidence = 0.0f;
    for (int i = 0; i < n_classes; i++) {
        probs[i] /= sum_exp;
        if (probs[i] > confidence) {
            confidence = probs[i];
            class_idx  = i;
        }
    }

    float confidence_pct = confidence * 100.0f;

    /* ── 10. Update shared result state ──────────────────────────────── */
    strncpy(last_label, ESC50_LABELS[class_idx], sizeof(last_label) - 1);
    last_label[sizeof(last_label) - 1] = '\0';
    last_confidence = confidence_pct;

    /* ── 11. Debug output to USB ─────────────────────────────────────── */
    Serial.printf("Predicted: %s (%.1f%%)\n", last_label, last_confidence);

    server.handleClient();
    delay(500);
}
