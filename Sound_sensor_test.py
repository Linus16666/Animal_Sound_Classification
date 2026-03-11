import serial
import matplotlib.pyplot as plt
import sys
import threading
from datetime import datetime


try:
    serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

sound_values = []
num_samples = 50000

print("Starting to collect sound values")


class sound_capturing:
    def __init__(self, serial_port, baud_rate, num_samples=50000, sound_values=None):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.num_samples = num_samples
        self.sound_values = sound_values if sound_values is not None else []

    def start_capture(self):
        print("Starting sound capture...")
        while len(self.sound_values) < self.num_samples:
            if self.serial_port.in_waiting > 0:
                try:
                    raw = self.serial_port.readline().decode('utf-8', errors='ignore')
                    tokens = [t for t in raw.replace('\r', '\n').split('\n') if t.strip()]
                    if tokens:
                        line = tokens[0].strip()
                        sound_value = int(line)
                        self.sound_values.append(sound_value)
                except ValueError as e:
                    print(f"Error parsing sound value: {e}")
                    pass
        print("Done collecting sound values for this period")
        return self.sound_values if self.sound_values else None
    
    def capture_samples(self):
        import time
        self._capture_start = time.time()
        while len(self.sound_values) < self.num_samples:
            if self.serial_port.in_waiting > 0:
                try:
                    raw = self.serial_port.readline().decode('utf-8', errors='ignore')
                    tokens = [t for t in raw.replace('\r', '\n').split('\n') if t.strip()]
                    if tokens:
                        sound_value = int(tokens[0].strip())
                        if 0 <= sound_value <= 1023:   # valid 10-bit ADC range
                            self.sound_values.append(sound_value)
                except ValueError:
                    pass
        self._capture_end = time.time()

    def get_actual_sample_rate(self):
        elapsed = self._capture_end - self._capture_start
        return int(round(len(self.sound_values) / elapsed)) if elapsed > 0 else 44100
    
    def stop(self):
        self.is_running = False
        self.serial_port.close()
        print("Sound capture stopped.")

    def read_and_print(self):
        self.is_running = True
        def read_loop():
            while self.is_running:
                print("Starting sound capture...")
                if self.serial_port.in_waiting > 0:
                    try:
                        raw = self.serial_port.readline().decode('utf-8', errors='ignore')
                        tokens = [t for t in raw.replace('\r', '\n').split('\n') if t.strip()]
                        if tokens:
                            line = tokens[0].strip()
                            sound_value = int(line)
                            if 0 <= sound_value <= 1023:
                                self.sound_values.append(sound_value)
                    except ValueError as e:
                        print(f"Error parsing sound value: {e}")
                        pass

        thread = threading.Thread(target=read_loop)
        thread.daemon = True
        thread.start()
        print("Sound capture started in a separate thread.")




import time
def capture():
    # Tell Arduino to start sending ADC data (needed for repeat runs)
    serial_port.write(b"START\n")
    serial_port.flush()
    sound_capture = sound_capturing(serial_port, 115200, num_samples)
    sound_capture.capture_samples()
    sound_captured = sound_capture.sound_values.copy()
    actual_rate = sound_capture.get_actual_sample_rate()
    # Stop ADC data — this drains the TX buffer before Python does any processing,
    # so the RESULT command sent later is received cleanly with no buffer backup.
    serial_port.write(b"STOP\n")
    serial_port.flush()
    print(f"Captured {len(sound_captured)} samples at ~{actual_rate} Hz")
    return sound_captured, actual_rate

def send_label(display_text, confidence_pct):
    """Send prediction result to Arduino LCD.
    Sends two separate commands: L1: for row 0, L2: for row 1.
    Arduino has stopped sending ADC data (STOP was sent after capture).
    """
    # Pad to 16 chars so any shorter text fully overwrites the previous row content
    serial_port.write(f"L1:{display_text:<16}\n".encode('utf-8'))
    serial_port.write(f"L2:{confidence_pct:.0f}%{'':<13}\n".encode('utf-8'))
    serial_port.flush()
    """
    plt.xlabel("Sample Number")
    plt.ylabel("Sound Value")
    plt.title("Sound Values Captured")
    plt.plot(range(len(sound_capture.sound_values)), sound_capture.sound_values)
    plt.show()
    print("plot")
    
    """
"""ja 
def main():
    sound_capture = sound_capturing(serial_port, 115200, num_samples, sound_values)
    sound_capture.read_and_print()
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Sound Value")
    ax.set_title("Real-time Sound Capture")
    try:
        while True:
            data_copy = sound_capture.sound_values.copy()
            line.set_data(range(len(data_copy)), data_copy)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Capture interrupted by user.")
    finally:
        sound_capture.stop()
        plt.ioff()
        plt.show()
        print("Sound capture stopped and plot closed.")
""" 



