"""
LCD debug test.
Upload the sketch first, then run:  python3 test_lcd.py

Reads back DBG: echo lines so we can see exactly what the Arduino received.
"""
import serial, time, sys

PORT = '/dev/ttyACM0'
BAUD = 115200

print(f"Opening {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.05)
except serial.SerialException as e:
    sys.exit(f"ERROR: {e}")

def drain(seconds):
    """Read and discard everything for `seconds` seconds."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if ser.in_waiting:
            ser.read(ser.in_waiting)

def read_lines(seconds):
    """Collect all lines received within `seconds` seconds."""
    lines = []
    deadline = time.time() + seconds
    buf = b""
    while time.time() < deadline:
        if ser.in_waiting:
            buf += ser.read(ser.in_waiting)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines.append(line.decode("utf-8", errors="ignore").strip())
    return lines

# Drain while Arduino boots — keeps Serial.println() from blocking
print("Draining 2 s while Arduino boots...")
drain(2.0)

# Send STOP and keep draining so Arduino can process it
print("Sending STOP...")
ser.write(b"STOP\n")
ser.flush()
drain(0.5)

# Send the two row commands
print("Sending L1/L2...")
ser.write(b"L1:Test Label\n")
ser.write(b"L2:99%\n")
ser.flush()

# Read back for 1 s — look for DBG: echo
print("Waiting for echo from Arduino...")
lines = read_lines(1.0)

dbg = [l for l in lines if l.startswith("DBG:")]
adc = [l for l in lines if not l.startswith("DBG:")]

print()
if dbg:
    print(f"Arduino echoed back: {dbg}")
    print("  → message WAS received. If LCD still blank, it is a hardware wiring issue.")
else:
    print("No DBG echo received.")
    print(f"  Got {len(adc)} ADC line(s): {adc[:3]}")
    print("  → STOP was not processed — Arduino was blocked. Try re-uploading the sketch.")

print()
print("Check LCD now — should show 'Test Label' / '99%'")
ser.close()
