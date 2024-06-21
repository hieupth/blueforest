import cv2
import faceparser
from PIL import Image

img = cv2.imread('resources/testface.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = faceparser.parse(Image.fromarray(img))
print(res)