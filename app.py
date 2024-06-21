import cv2
import os
import numpy as np
import gradio as gr
import faceparser
import imgutils
from PIL import Image
from fastapi import FastAPI

CUSTOM_PATH = os.environ.get('CUSTOM_PATH', '')

tree_img = cv2.cvtColor(cv2.imread('resources/background.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
sky_img = cv2.cvtColor(cv2.imread('resources/skinground.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def fusion_step_1(sky, tree, skin):
  skin_rgb = cv2.cvtColor(skin, cv2.COLOR_GRAY2RGB)
  dst1 = cv2.bitwise_and(sky, skin_rgb, mask=skin)
  dst2 = cv2.bitwise_and(tree, cv2.bitwise_not(skin_rgb), mask=cv2.bitwise_not(skin))
  return dst1, dst2

def effect1(image, target, mask, bgmask, brightness):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray = imgutils.brightness(gray, 0, brightness)
  gray = cv2.bitwise_and(gray, mask, mask=mask)
  gray[gray == 0] = 255
  #return gray
  rate = gray / 255
  y, cr, cb = cv2.split(cv2.cvtColor(target, cv2.COLOR_RGB2YCrCb))
  z = np.array(np.multiply(y, rate), dtype=np.uint8)
  res = cv2.cvtColor(cv2.merge((z, cr, cb)), cv2.COLOR_YCR_CB2RGB)
  return res

def run3(image, brightness):
  tree = cv2.resize(tree_img, (image.shape[1], image.shape[0]))
  sky= cv2.resize(sky_img, (image.shape[1], image.shape[0]))
  masks = faceparser.parse(Image.fromarray(image))
  otsu = imgutils.otsu(image)
  #
  mask = cv2.bitwise_and(otsu, cv2.bitwise_not(masks[0]))
  mask_fixed = cv2.bitwise_and(mask, cv2.bitwise_not(masks[13] + masks[18]))
  mask_color_adjust = mask_fixed + masks[0]
  dst1, dst2 = fusion_step_1(sky, tree, mask_fixed)
  #
  dst3 = effect1(image, dst2, cv2.bitwise_not(mask_color_adjust), masks[0], brightness)
  return dst3 + dst1

demo = gr.Interface(
  fn=run3,
  inputs=["image", gr.Slider(value=0, minimum=-127, maximum=127, step=1)],
  outputs=["image"]
)
# demo.launch().queue(default_concurrency_limit=15)

app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is your main app"}

#io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)