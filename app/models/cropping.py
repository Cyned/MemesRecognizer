import cv2
import math

import numpy as np
from PIL import Image

from models.pre_processor import Tesseract


tesseract = Tesseract()


def read_coord(data):
    data = [coord.strip().split(',') for coord in data.split('\r\n')]
    data = [[[int(line[n - 1]), int(el)] for n, el in enumerate(line) if n % 2 == 1] for line in data]
    return data


def crop_rectangle(coord_list):
    x1 = max(min([i[0] for i in coord_list]), 0)
    x2 = max([j[0] for j in coord_list])

    y1 = max(min([j[1] for j in coord_list]), 0)
    y2 = max([j[1] for j in coord_list])
    return x1, y1, x2, y2


def angle(coor):
    x1, y1 = coor[0]
    x2, y2 = coor[1]
    x3, y3 = coor[2]

    d1 = distance(x1, y1, x2, y2)
    d2 = distance(x2, y2, x3, y3)

    [x1, y1, x2, y2] = [x1, y1, x2, y2] if d1 > d2 else [x2, y2, x3, y3]
    b = x2 - x1
    a = y2 - y1

    if a == 0:
        return 0
    if math.atan(b / a) <= -1:
        return 270 - math.atan(b / a) * 180 / math.pi
    return 90 - math.atan(b / a) * 180 / math.pi


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def distance(x1, y1, x2, y2):
    d = math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
    return math.sqrt(d)


def rotation_image(image, coordinate):
    # image_original = cv2.imread(filepath)
    alpha = angle(coordinate)

    (x1, y1, x2, y2) = crop_rectangle(coordinate)
    crop = image[y1:y2, x1:x2]
    try:
        img = rotateImage(crop, alpha)
    except Exception:
        return crop
    else:
        return img


def run_to_crop(path_to_img, coordinate, path_to_res):
    image = cv2.imread(path_to_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = rotation_image(image, coordinate)
    image = tesseract.preprocess(image)
    res = Image.fromarray(image, 'RGB')
    res.save(path_to_res)
