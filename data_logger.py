import time
import serial
import re
import subprocess

def main(args):
    return 0
    
def replace_time(text, timestamp):
    pattern = r"^([^,]*)"
    return re.sub(pattern, str(timestamp), text)

    
if __name__ == '__main__':
    import sys
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    ser.reset_input_buffer()

    while True:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        filename = "log_data_" + current_time + ".csv"
        file_ts = open(filename, "w")
        file_ts.write("Time,ECG,PCG,PPG\n")
        file = open("log_data.csv","w")
        file.write("Time,ECG,PCG,PPG\n")
        count = 0

        while count < 900000: # 60000 = 1 minute (60 seconds)
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip().replace(" ",",")
                data = replace_time(line, count)
                file.write(data + "\n")
                file_ts.write(data + "\n")
                print(data)
                count += 10
        file.close()
        file_ts.close()
        subprocess.Popen(["python3", "run_logging_display.py"])
    time.sleep(0.1)
