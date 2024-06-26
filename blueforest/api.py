from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import numpy as np
import io
import os
from blueforest.app import run
from pillow_heif import register_heif_opener
import hashlib
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary import CloudinaryImage
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
register_heif_opener()

@app.post("/run/")
async def create_upload_file(file: UploadFile, caption: str = ""):
  image = await file.read(-1)
  image = Image.open(io.BytesIO(image))
  image = image.convert('RGB')
  res = run(np.array(image, dtype=np.uint8), caption)
  img_byte_arr = io.BytesIO()
  res.save(img_byte_arr, format='JPEG')
  img_byte_arr = img_byte_arr.getvalue()
  return Response(content=img_byte_arr, media_type="image/jpeg")

@app.post("/run-url")
async def create_upload_file(file: UploadFile, caption: str = ""):
  image = await file.read(-1)
  image = Image.open(io.BytesIO(image))
  image = image.convert('RGB')
  res = run(np.array(image, dtype=np.uint8), caption)
  img_byte_arr = io.BytesIO()
  res.save(img_byte_arr, format='JPEG')
  # img_byte_arr = img_byte_arr.getvalue()
  hash = hashlib.md5(img_byte_arr.getbuffer()).hexdigest()
  cloudinary.uploader.upload_image(img_byte_arr.getbuffer(), public_id=hash, unique_filename = False, overwrite=True)
  srcURL = CloudinaryImage(hash).build_url()
  return JSONResponse({'url': srcURL})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)