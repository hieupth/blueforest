import os
import cv2 
import random
from . import imgutils
from . import faceparser
from PIL import Image

tree = cv2.cvtColor(cv2.imread(os.environ.get('TREE_IMAGE', os.path.join(os.getcwd(), 'resources', 'background.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
sky = cv2.cvtColor(cv2.imread(os.environ.get('SKY_IMAGE', os.path.join(os.getcwd(), 'resources', 'skinground.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

captions = []
captions_dir = os.environ.get('CAPTION_IMAGES_DIR', os.path.join(os.getcwd(), 'resources', 'caption'))
for i in range(1, 6):
  cap = os.path.join(captions_dir, f'{i}.png')
  cap = Image.open(cap)
  captions.append(cap)

def run(image, caption = None):
  # Copy tree and sky images.
  _tree = tree.copy()
  _sky = cv2.resize(sky, (_tree.shape[1], _tree.shape[0]))
  # Crop portait and make sure portait width smaller than tree width when scaled height to tree height.
  portait, coor = imgutils.crop_portait(image)
  portait, ratio = imgutils.scale_height(portait, tree.shape[0])
  portait = imgutils.crop_to_fit_width(portait, _tree.shape[1], left=False)
  # Get face parsing mask and otsu.
  masks = faceparser.parse(Image.fromarray(portait))
  otsu, thresh = imgutils.otsu(portait)
  # Get masks.
  mask = cv2.bitwise_and(otsu, cv2.bitwise_not(masks[0]))
  mask_fixed = cv2.bitwise_and(mask, cv2.bitwise_not(masks[13] + masks[18]))
  mask_color_adjust = mask_fixed + masks[0]
  #
  x_offset = _tree.shape[0] / 100 * 0
  x_margin_right = _tree.shape[1] - portait.shape[1] - x_offset
  x_margin_right = int(max(x_margin_right, 0))
  #
  portait_tree = _tree[0:_tree.shape[0], x_margin_right:(x_margin_right+portait.shape[1])]
  portait_sky = _sky[0:_tree.shape[0], x_margin_right:(x_margin_right+portait.shape[1])]
  #
  dst1, dst2 = imgutils.bitwisepair(portait_sky, portait_tree, mask_fixed)
  dst3 = imgutils.brightness_by_grayscale(portait, dst2, cv2.bitwise_not(mask_color_adjust), thresh=thresh, brightness=0)
  #
  merged = imgutils.draw_image_on_image(dst3 + dst1, _tree, x_margin_right, 0)
  if caption is None:
    cap = random.choice(captions).copy()
    merged = Image.fromarray(merged).convert('RGBA')
    cap = cap.resize((merged.width, merged.height))
    merged.paste(cap, (0, 0), cap)
    merged = merged.convert('RGB')
  return merged
