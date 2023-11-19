import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
from models import load_classes

# classes = load_classes("data/coco.names")
# cls2idx = {cls: i for i, cls in enumerate(classes)}

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def plot_boxes(img_path, label_path, classes):
    """
    This is modified from eriklindernoren's yolov3: https://github.com/eriklindernoren/PyTorch-YOLOv3
    
    eriklindernoren's `detect.py` use `plt` to plot text so that cleaner
    """
    # create plot
    img = np.array(Image.open(img_path).convert('RGB'))  # (h,w,c)
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img)
    
    # read ground-turth boxes
    boxes = None
    if os.path.exists(label_path):
        boxes = torch.from_numpy(np.loadtxt(open(label_path)).reshape(-1, 5))
        boxes[:, 1:] = xywh2xyxy(boxes[:, 1:])
        boxes[:, 1] *= img.shape[1]
        boxes[:, 2] *= img.shape[0]
        boxes[:, 3] *= img.shape[1]
        boxes[:, 4] *= img.shape[0]
        boxes = np.round(boxes)
        
    # Bounding-box colors
    random.seed(0)
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        box_w = x2 - x1
        box_h = y2 - y1

        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=colors[int(cls)], facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=classes[int(cls)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(cls)], "pad": 0},
            fontsize=10,
        )
    
    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
#     filename = path.replace("\\", "/").split("/")[-1].split(".")[0]
#     plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
#     plt.close()
    plt.show()
