import os
import cv2 
import random
from . import imgutils
from . import faceparser
from PIL import Image, ImageFont, ImageDraw


tree = cv2.cvtColor(cv2.imread(os.environ.get('TREE_IMAGE', os.path.join(os.getcwd(), 'resources', 'background.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
sky = cv2.cvtColor(cv2.imread(os.environ.get('SKY_IMAGE', os.path.join(os.getcwd(), 'resources', 'skinground.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#
captions = []
captions_dir = os.environ.get('CAPTION_IMAGES_DIR', os.path.join(os.getcwd(), 'resources', 'caption'))
for i in range(1, 6):
  cap = os.path.join(captions_dir, f'{i}.png')
  cap = Image.open(cap)
  captions.append(cap)
#
customcap = Image.open(os.environ.get('CUSTOM_CAP_IMAGE', os.path.join(os.getcwd(), 'resources', 'customcaption.png')))
#
font = ImageFont.truetype(os.environ.get('FONT', os.path.join(os.getcwd(), 'resources', 'fonts', 'SVN-GOTHAM BOLD.TTF')), 70)


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
  merged = Image.fromarray(merged).convert('RGBA')
  if caption is None or len(caption) < 1:
    cap = random.choice(captions).copy()
    cap = cap.resize((merged.width, merged.height))
    merged.paste(cap, (0, 0), cap)
  else:
    _caps = caption.split(' ')
    _cap = ""
    for i in range(0, len(_caps)):
      z = (i + 1) % 2
      if z == 0:
        _cap = f'{_cap} {_caps[i]}\n'
      else:
        _cap = f'{_cap}{_caps[i]}'
    _cap = _cap.upper()
    _caps = _cap.splitlines()
    cap = customcap.resize((merged.width, merged.height))
    merged.paste(cap, (0, 0), cap)
    h, w = merged.height, merged.width
    for i in range(0, len(_caps)):
      draw = ImageDraw.Draw(merged)
      draw.text((50, int(h * 0.6 + i * 80)), _caps[i], (255, 255, 255), font=font)
  merged = merged.convert('RGB')
  # merged.show()
  return merged
