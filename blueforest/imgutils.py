import cv2
import math
import numpy as np
from .facedetect import facedet
import matplotlib.pyplot as plt

def otsu(image, type=np.uint8):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  thresh, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return np.array(otsu, dtype=type), thresh

def adjust_brightness(image, contrast, brightness):
  img = np.int16(image)
  img = img * (contrast/127+1) - contrast + brightness
  img = np.clip(img, 0, 255)
  img = np.uint8(img) 
  return img

def bitwisepair(src1, src2, mask):
  maskrgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
  dst1 = cv2.bitwise_and(src1, maskrgb, mask=mask)
  dst2 = cv2.bitwise_and(src2, cv2.bitwise_not(maskrgb), mask=cv2.bitwise_not(mask))
  return dst1, dst2

def crop_portait(image):
  crops = []
  bboxes, scores, _, _ = facedet.inference(image, det_thres=0.4, get_layer='face')
  if len(bboxes) > 0:
    for xyxy, _ in zip(bboxes, scores):
      x1, y1, x2, y2 = xyxy.astype(int)
      h, w = int(math.fabs(y2 - y1)), int(math.fabs(x2 - x1))
      xx1 = x1 - 0.8 * w
      xx1 = max(xx1, 0)
      xx2 = x2 + 0.5 * w
      xx2 = min(xx2, image.shape[1])
      yy1 = y1 - 0.35 * h
      yy1 = max(yy1, 0)
      yy2 = y2 + 0.35 * h
      yy2 = min(yy2, image.shape[0])
      coor = np.array([xx1, yy1, xx2, yy2], dtype=int)
      crops.append(coor)
  coor = crops[0]
  cropped = image[coor[1]:coor[3], coor[0]:coor[2]]
  return cropped, coor

def scale_height(image, height):
  size = image.shape[:2]
  ratio = height / size[0]
  return cv2.resize(image, (int(size[1] * ratio), int(height))), ratio

def brightness_by_grayscale(image, target, mask, thresh = None, brightness = 0, constrast = 0):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray = adjust_brightness(gray, constrast, brightness)
  gray = cv2.bitwise_and(gray, mask, mask=mask)
  if thresh is not None:
    gray[gray > thresh] = 255 - gray[gray > thresh]
  gray[gray == 0] = 255
  rate = gray / 255
  y, cr, cb = cv2.split(cv2.cvtColor(target, cv2.COLOR_RGB2YCrCb))
  z = np.array(np.multiply(y, rate), dtype=np.uint8)
  res = cv2.cvtColor(cv2.merge((z, cr, cb)), cv2.COLOR_YCR_CB2RGB)
  return res

def draw_image_on_image(image, background, x_offset, y_offset):
  y1, y2 = y_offset, y_offset + image.shape[0]
  x1, x2 = x_offset, x_offset + image.shape[1]
  background[y1:y2, x1:x2] = image
  return background

def crop_to_fit_width(image, width, left=True):
  w = image.shape[1]
  if left:
    res = image[0:h, 0:min(w, width)]
  else:
    start = w - min(w, width)
    start = start if start > 0 else 0
    h = image.shape[0]
    res = image[0:h, start:w]
  return res
