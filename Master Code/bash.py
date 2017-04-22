import PIL
from PIL import Image

img = Image.open('output_image.png')

basewidth = 512

hsize = 512

img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

img.save('/home/aakash-sinha/Documents/Aspectus/Image Segmentation/tensorflow_notes-master/outputResized/resized_image.png')
