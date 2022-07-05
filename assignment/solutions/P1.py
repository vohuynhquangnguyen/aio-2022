import os
import cv2
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
plt.rcParams["figure.figsize"] = [10,10]

random.seed(42)
num_classes = 50
char_dig = string.ascii_lowercase + ''.join([str(c) for c in range(10)]) + "&"
CHAR_COLOR = {c:[random.randint(0, 255) for _ in range(3)] for c in char_dig}
CHAR_COLOR['others'] = [0,0,0]
# COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]

def draw_one_box(box, image, label, color, line_thickness=None):
    # Plots one bounding box on image img
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_one_poly(pts, raw_pts, image, label, color, line_thickness=None, isClosed=True):
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    cv2.polylines(image, [pts], isClosed, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        c1 = tuple(map(int, raw_pts[0]))
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_one_img(img_path, data, is_from_xml):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if is_from_xml:
        for class_name, bbox in data:
            draw_one_box(box=bbox, image=image, label=class_name, color=CHAR_COLOR.get(class_name.lower(), [0, 0, 0]))
    else:
        for class_name, pts in data:
            pts = np.array(pts, np.int32)
            raw_pts = pts.copy()
            pts = pts.reshape((-1, 1, 2))
            draw_one_poly(pts=pts, raw_pts=raw_pts, image=image, label=class_name, color=CHAR_COLOR.get(class_name.lower(), [0, 0, 0]))
    
    if image is None:
        raise Exception("Image is None")
    
    plt.imshow(image)