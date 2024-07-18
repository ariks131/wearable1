# TODO-DONE: capture data form arduino serial to csv format 
import serial
import re 

def main(args):
    return 0
    
def replace_time(text, timestamp):
    pattern = r"^([^,]*)"
    return re.sub(pattern, str(timestamp), text)
    
if __name__ == '__main__':
    import sys
    #sys.exit(main(sys.argv))
    print("test")
    #ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
	
    ser.reset_input_buffer()
    print(ser)
    line = ser.readline().decode('utf-8').rstrip()
    print(line)

