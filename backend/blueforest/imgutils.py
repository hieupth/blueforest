import cv2
import numpy as np

def otsu(image, type=np.uint8):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return np.array(otsu, dtype=type)

def brightness(image, contrast, brightness):
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