import serial
import matplotlib.pyplot as plt
import sys
import threading
from datetime import datetime


try:
    serial_port = serial.Serial('COM6', 115200, timeout=1)
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
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        sound_value = int(line)
                        self.sound_values.append(sound_value)
                except ValueError as e:
                    print(f"Error parsing sound value: {e}")
                    pass
        print("Done collecting sound values for this period")
        return self.sound_values if self.sound_values else None
    
    def capture_samples(self):
        if self.serial_port.in_waiting > 0:
                    for _ in range(self.num_samples - len(self.sound_values)):
                        try:
                            line = self.serial_port.readline().decode('utf-8').strip()
                            if line:
                                sound_value = int(line)
                                if sound_value < 10000 and sound_value > 200: #maybe delete?
                                    self.sound_values.append(sound_value)
                                else:
                                    pass
                                if len(self.sound_values) >= self.num_samples:
                                    print("Collected enough sound values, returning")
                                    print(len(self.sound_values))
                                    #self.sound_values.pop(0)
                        except ValueError as e:
                            print(f"Error parsing sound value: {e}")
                            pass
    
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
                    for _ in range(self.num_samples - len(self.sound_values)):
                        try:
                            line = self.serial_port.readline().decode('utf-8').strip()
                            if line:
                                sound_value = int(line)
                                if sound_value < 10000 and sound_value > 200:
                                    self.sound_values.append(sound_value)
                                else:
                                    pass
                                if len(self.sound_values) >= self.num_samples:
                                    print("Collected enough sound values, returning")
                                    print(len(self.sound_values))
                                    #self.sound_values.pop(0)
                        except ValueError as e:
                            print(f"Error parsing sound value: {e}")
                            pass

        thread = threading.Thread(target=read_loop)
        thread.daemon = True
        thread.start()
        print("Sound capture started in a separate thread.")




import time
def main():
    sound_capture = sound_capturing(serial_port, 115200, num_samples, sound_values)
    sound_capture.capture_samples()
    sound_captured = sound_capture.sound_values.copy()
    plt.xlabel("Sample Number")
    plt.ylabel("Sound Value")
    plt.title("Sound Values Captured")
    plt.plot(range(len(sound_captured.sound_values)), sound_captured.sound_values)
    plt.show()
    return sound_captured
"""
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



