from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1107
from time import sleep
from PIL import Image, ImageDraw, ImageFont
import os
import time

serial = i2c(port=1, address=0x3C)
device = sh1107(serial, rotate=1, width=128)

# dev-res sh1107: 128x128
# ~ with canvas(device) as draw:
    # ~ draw.text((0, 0), "Hello World", fill="white")
# ~ sleep(10)


font_path = os.path.abspath("firacode.ttf")
font_large = ImageFont.truetype(font_path,size=32)
font = ImageFont.truetype(font_path,size=16)

# Create an image with a black background
image = Image.new("1", (device.width, device.height))

# Create a drawing context
draw = ImageDraw.Draw(image)
y_pos = 10

with open("result_display.txt", "r") as file:
	data_received = file.read()
print(data_received)
sbp, dbp, status = data_received.split(",")
sbp = int(float(sbp))
dbp = int(float(dbp))
status = status.split()

x1, y1 = 0, 0
x2, y2 = device.width - 1, device.height - 1
draw.rectangle([(x1, y1), (x2, y2)], fill=0)
# draw.text(x,y) -> (0,0) top-left
#draw.text((0, y_pos), str(sbp) + "/" + str(dbp), font=font_large, fill=1)
draw.text((0, y_pos), "Process" , font=font_large, fill=1)
draw.text((0, y_pos+40), "logging", font=font_large, fill=1)
if len(status) > 1: draw.text((0, y_pos+56), status[1], font=font, fill=1)	
# Display the image on the OLED screen
device.display(image)

time.sleep(10)



