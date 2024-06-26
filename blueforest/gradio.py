# from blueforest.imgutils import *
from blueforest.app import run
import gradio as gr
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import io

register_heif_opener()

def run3(image, caption):
  #img_byte_arr = io.BytesIO()
  #image.save(img_byte_arr, format='PNG')
  #img_byte_arr = img_byte_arr.getvalue()
  #img = Image.open(img_byte_arr)
  #img.show()
  img = image
  img = np.array(img, dtype=np.uint8)
  return run(img, caption)

demo = gr.Interface(
  fn=run3,
  inputs=[gr.Image(type='pil'), "text"],
  outputs=["image"]
)
demo.queue(default_concurrency_limit=15).launch(server_name="0.0.0.0")