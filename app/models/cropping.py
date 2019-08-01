import cv2
import numpy as np
import math
import os

from PIL import Image


def read_coord(file_coordinates):
    with open(file_coordinates, 'r') as f:
        data = f.readlines()
    data = [coord.strip().split(',') for coord in data]
    data = [[[line[n - 1], el] for n, el in enumerate(line) if n % 2 == 1] for line in data]
    return data


def tool(image, coordinates):
    pts = np.array(coordinates, dtype=np.int32)
    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, pts, 3)
    mask = mask.astype(np.bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]

    img = image
    # Find centroid of polygon
    (meanx, meany) = pts.mean(axis=0)
    (cenx, ceny) = (img.shape[1] / 2, img.shape[0] / 2)

    # Make integer coordinates for each of the above
    (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)

    # Calculate final offset to translate source pixels to centre of image
    (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

    # Define remapping coordinates
    (mx, my) = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    ox = (mx - offsetx).astype(np.float32)
    oy = (my - offsety).astype(np.float32)

    # Translate the image to centre
    out_translate = cv2.remap(out, ox, oy, cv2.INTER_LINEAR)

    # Determine top left and bottom right of translated image
    topleft = pts.min(axis=0) + [offsetx, offsety]
    bottomright = pts.max(axis=0) + [offsetx, offsety]
    return out_translate


def crop_rectangle(coord_list):
    x1 = min([i[0] for i in coord_list])
    x2 = max([j[0] for j in coord_list])

    y1 = min([j[1] for j in coord_list])
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


def read_coord(data):  # file_coordinates
    data = [d.split(',') for d in data.split()]
    return [[[int(line[n - 1]), int(el)] for n, el in enumerate(line) if n % 2 == 1] for line in data]


def rotate_img(image, angle):
    # calculate the center of he image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (h, w))
    return rotated


def distance(x1, y1, x2, y2):
    d = math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
    return math.sqrt(d)


def rotation_image(image, coordinate):
    alpha = angle(coordinate)
    croped = tool(image, coordinate)

    img = rotate_img(croped, alpha)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = img[y:y + h, x:x + w]
    return crop


def run_to_crop(path_to_img, coordinate, path_to_res):
    image = cv2.imread(path_to_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = rotation_image(image, coordinate)
    res = Image.fromarray(image, 'RGB')

    my_path = os.path.dirname(path_to_res)
    if not os.path.exists(my_path):
        os.makedirs(my_path)

    res.save(path_to_res)
