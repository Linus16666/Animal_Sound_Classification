char inputBuffer[32];
int  bufLen     = 0;
bool sendingADC = true;

void processIncoming() {
  if (strcmp(inputBuffer, "STOP") == 0) {
    sendingADC = false;
  } else if (strcmp(inputBuffer, "START") == 0) {
    sendingADC = true;
  }
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      inputBuffer[bufLen] = '\0';
      processIncoming();
      bufLen = 0;
    } else if (c != '\r' && bufLen < 31) {
      inputBuffer[bufLen++] = c;
    }
  }

  if (sendingADC) {
    int sound = analogRead(A1);
    Serial.println(sound);
  }
}
