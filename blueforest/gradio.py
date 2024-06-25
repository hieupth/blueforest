# from blueforest.imgutils import *
from blueforest.app import run
import gradio as gr

def run3(image, caption):
  return run(image, caption)

demo = gr.Interface(
  fn=run3,
  inputs=["image", "text"],
  outputs=["image"]
)
demo.launch().queue(default_concurrency_limit=15)