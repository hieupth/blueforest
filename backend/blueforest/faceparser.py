import cv2
import torch
from PIL import Image
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np

# convenience expression for automatically determining device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)


def parse(image, seperated = True):
    # run inference on image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)
    # resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(logits,
                    size=image.size[::-1], # H x W
                    #size=(image.shape[0], image.shape[1]),
                    mode='bilinear',
                    align_corners=False)
    # get label masks
    labels = upsampled_logits.argmax(dim=1)[0]
    # move to CPU to visualize in matplotlib
    labels = labels.cpu().numpy()
    # return
    if not seperated:
        return labels
    else:
        masks = []
        for i in range(0, 19):
            _logit = i + 1
            x = labels.copy() + 1
            x[x != _logit] = 0
            x[x == _logit] = 255
            x = np.array(x, dtype=np.uint8)
            masks.append(x)
        return masks