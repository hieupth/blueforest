import os
import cv2 
from . import imgutils
from . import faceparser
from PIL import Image

tree = cv2.cvtColor(cv2.imread(os.environ.get('TREE_IMAGE', os.path.join(os.getcwd(), 'resources', 'background.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
sky = cv2.cvtColor(cv2.imread(os.environ.get('SKY_IMAGE', os.path.join(os.getcwd(), 'resources', 'skinground.jpg')), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def run(image):
  portait, coor = imgutils.crop_portait(image)
  portait, ratio = imgutils.scale_height(portait, tree.shape[0])
  _sky = cv2.resize(sky, (tree.shape[1], tree.shape[0]))
  masks = faceparser.parse(Image.fromarray(portait))
  otsu, thresh = imgutils.otsu(portait)
  #
  mask = cv2.bitwise_and(otsu, cv2.bitwise_not(masks[0]))
  mask_fixed = cv2.bitwise_and(mask, cv2.bitwise_not(masks[13] + masks[18]))
  mask_color_adjust = mask_fixed + masks[0]
  #
  x_offset = tree.shape[0] / 100 * 5
  portait_tree_coor = [int(tree.shape[1] - portait.shape[1] - x_offset), 0]
  portait_tree = tree[0:tree.shape[0], portait_tree_coor[0]:portait_tree_coor[0]+portait.shape[1]]
  portait_sky = _sky[0:tree.shape[0], portait_tree_coor[1]:portait_tree_coor[1]+portait.shape[1]]
  #
  dst1, dst2 = imgutils.bitwisepair(portait_sky, portait_tree, mask_fixed)
  dst3 = imgutils.brightness_by_grayscale(portait, dst2, cv2.bitwise_not(mask_color_adjust), thresh=thresh, brightness=0)
  #
  merged = imgutils.draw_image_on_image(dst3 + dst1, tree, portait_tree_coor[0], 0)
  cv2.imshow('test', cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
  cv2.waitKey()

