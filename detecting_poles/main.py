from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import torch
import torch.utils.data
import torchvision.transforms as T


def extract_coordinates(file):
    """ Extract [xmin, ymin, xmax, ymax] coordinates from XML file """
    tree = ET.parse(file)
    root = tree.getroot()
    boxes = []
    for bb in root.iter('bndbox'):
        x0 = int(bb[0].text)  # xmin
        y1 = int(bb[3].text)  # ymax

        x1 = int(bb[2].text)  # xmax
        y0 = int(bb[1].text)  # ymin
        xmin = min(x0, x1)
        xmax = max(x0, x1)
        ymin = min(y0, y1)
        ymax = max(y0, y1)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def draw_bounding_box(image, corners_up_left, corners_bottom_right):
    """ Draw bounding boxes given the corners [xmin, ymax] and [xmax, ymin] with CV2 """
    for i in range(len(corners_up_left)):
        x = corners_up_left[i]
        y = corners_bottom_right[i]  # x and y are two opposite corners of the rectangle
        cv.rectangle(image, x, y, (255, 255, 255), thickness=10)  # (0, 255, 0), 2)
    plt.figure()
    plt.imshow(image)  # , cmap="gray")
    plt.show()


def display_image_with_bounding_boxes(image, target, to_npy=True):
    """ Display image with bounding boxes given image and target from dataset with CV2 """
    if to_npy:
        image = image.numpy()
    image = np.sum(image, 0)

    # bounding boxes
    boxes = target['boxes']
    boxes = boxes.cpu()
    if type(boxes) != np.ndarray:
        boxes = boxes.numpy()
    num_boxes = boxes.shape[0]
    corners_up_left = [(boxes[i][0], boxes[i][3]) for i in range(num_boxes)]
    corners_bottom_right = [(boxes[i][2], boxes[i][1]) for i in range(num_boxes)]
    draw_bounding_box(image, corners_up_left, corners_bottom_right)


def PIL_disp_img_bb(image, target, color='red'):
    """ Creates a PIL.Image of the given :image: with the bounding boxes contained in :target:"""
    to_pil = T.ToPILImage(mode='RGB')
    base = to_pil(image)
    base = base.convert('RGBA')
    box_container = Image.new(base.mode, base.size, (255,255,255, 0))
    d = ImageDraw.Draw(box_container)
    boxes = target['boxes']
    boxes = boxes.cpu()
    if type(boxes) != np.ndarray:
        boxes = boxes.numpy()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        bb = d.rectangle(boxes[i], outline=color)
    base.show()
    box_container.show()
    # return base, box_container
    out = Image.alpha_composite(base, box_container)
    return out


class AirportDataSet(object):
    def __init__(self, root_dir, transforms):
        self.root = root_dir
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.images = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.bb = list(sorted(os.listdir(os.path.join(root_dir, "annotations"))))

    def __getitem__(self, index):
        # load images and bounding boxes
        image_path = os.path.join(self.root, "images", self.images[index])
        bb_path = os.path.join(self.root, "annotations", self.bb[index])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = extract_coordinates(bb_path)

        # convert everything into a tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes), ), dtype=torch.int64)

        # image_id = torch.tensor([index])
        # area = (boxes[:, 3] - boxes[:, 1])*(boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        # target['area'] = area
        target['labels'] = labels
        target['image_id'] = torch.as_tensor(index, dtype = torch.int64)
        target['area'] = torch.zeros((len(boxes), ), dtype=torch.int64)
        target['iscrowd'] = torch.zeros((len(boxes), ), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            # transform target
            # boxes[:, 0] = width - boxes[:, 0]  # x_min
            # boxes[:, 2] = width - boxes[:, 2]  # x_max
            # target['boxes'] = boxes

        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    model = torch.load('/models/dp_faster_rcnn.pth')

    test_set = AirportDataSet('/dataset/test2/', None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pil2tensor = T.ToTensor()
    model.eval()
    for index in range(len(test_set)):
        test_img, _ = test_set[index]
        target = _
        test_img = pil2tensor(test_img)
        with torch.no_grad():
            pred = model([test_img.to(device)])
        out = PIL_disp_img_bb(test_img, pred[0])
        out.save('/results/test2/annotated'+str(index)+'.png')
        true = PIL_disp_img_bb(test_img, target, color=(0, 255, 0))
        true.save('/dataset/test2/annotated'+str(index)+'.png')

