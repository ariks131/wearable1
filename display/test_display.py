from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1107
from time import sleep
from PIL import Image, ImageDraw, ImageFont
import os
import time

serial = i2c(port=1, address=0x3C)
device = sh1107(serial, rotate=0, width=128)

# dev-res sh1107: 128x128
# ~ with canvas(device) as draw:
    # ~ draw.text((0, 0), "Hello World", fill="white")
# ~ sleep(10)
font_path = os.path.abspath("firacode.ttf")
font = ImageFont.truetype(font_path,size=16)

# Create an image with a black background
image = Image.new("1", (device.width, device.height))

# Create a drawing context
draw = ImageDraw.Draw(image)

while True:
    device.clear
    # Draw larger text on the image
    # draw.text(x,y) -> (0,0) top-left
    draw.text((10, 0), "Larger Text", font=font, fill=1)
    draw.text((10, 16), "Larger Text", font=font, fill=1)

    # Display the image on the OLED screen
    device.display(image)

    time.sleep(1)
