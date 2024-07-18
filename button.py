from time import sleep
from PIL import Image, ImageDraw, ImageFont
import os
import time
import serial
import re
import RPi.GPIO as GPIO

def replace_time(text, timestamp):
    pattern = r"^([^,]*)"
    return re.sub(pattern, str(timestamp), text)

BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

previous_button_state = GPIO.input(BUTTON_PIN)

try:
    while True:
        time.sleep(0.1)
        button_state = GPIO.input(BUTTON_PIN)
        if  button_state != previous_button_state:
            previous_button_state = button_state
            if button_state == GPIO.HIGH:
                print("Starting bio-signal measurement")
                ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
                    #ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)

                ser.reset_input_buffer()
                    
                    # TODO: Open csv file
                    # with open("log_data.csv","w", newline="") as file:
                    # writer = csv.writer(file)
                file = open("log_data.csv","w")
                file.write("Time,ECG,PCG,PPG\n")
                count = 0
                while count < 10000: # 60000 = 1 minute (60 seconds)
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8').rstrip().replace(" ",",")
                        data = replace_time(line, count)
                            # TODO: Write to csv file
                            #writer.writerow(data)
                        file.write(data + "\n")
                        print(data)
                            # ~ x1, y1 = 0, 0
                            # ~ x2, y2 = device.width - 1, device.height - 1
                            # ~ draw.rectangle([(x1, y1), (x2, y2)], fill=0)
                            # ~ # draw.text(x,y) -> (0,0) top-left
                            # ~ draw.text((0, y_pos), "Logging Data..", font=font_large, fill=1)
                            # ~ device.display(image)
                            # Close file after 1 minutes
                        count += 10
                file.close()
except KeyboardInterrupt:
    GPIO.cleanup()