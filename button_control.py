import RPi.GPIO as GPIO
import time
import subprocess

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Use the GPIO pin you connected the button to

program_running = False

def button_callback(channel):
    global program_running
    if program_running:
        print("Stopping program")
        subprocess.Popen(["python3", "stop_logging_display.py"])
        subprocess.run(["pkill", "-f", "data_logger.py"])  # Replace with your actual program name
    else:
        print("Starting program")
        subprocess.Popen(["python3", "start_logging_display.py"])
        subprocess.Popen(["python3", "data_logger.py"])  # Replace with your actual program name
    program_running = not program_running

GPIO.add_event_detect(26, GPIO.FALLING, callback=button_callback, bouncetime=300)

try:
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
