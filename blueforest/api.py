from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import numpy as np
import io
from blueforest.app import run

app = FastAPI()

@app.post("/run/")
async def create_upload_file(file: UploadFile, caption: str = ""):
  image = await file.read(-1)
  image = Image.open(io.BytesIO(image))
  res = run(np.array(image, dtype=np.uint8), caption)
  img_byte_arr = io.BytesIO()
  res.save(img_byte_arr, format='JPEG')
  img_byte_arr = img_byte_arr.getvalue()
  return Response(content=img_byte_arr, media_type="image/jpeg")