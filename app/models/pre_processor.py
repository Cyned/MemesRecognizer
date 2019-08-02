import cv2
import os

import numpy as np
from PIL import PngImagePlugin

from random import randint
from PIL import Image as ImagePIL

from typing import Union, Tuple
from config import DATA_DIR


# path to the preprocessor cache
CACHE_DIR: str = os.path.join(str(os.path.dirname(os.path.abspath(__file__))), 'preprocess_cache/')


class Preprocessor(object):

    def __init__(self, image: Union[str, np.array]):
        if isinstance(image, str):
            self.image = ImagePIL.open(image)
        elif isinstance(image, np.ndarray):
            self.image = ImagePIL.fromarray(image)
        else:
            raise TypeError('Incorrect image type')

        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        self.CACHED_IMAGE = os.path.join(CACHE_DIR, f'image{randint(1, 100000)}.png')
        self.image.save(self.CACHED_IMAGE)

    def to_contrast(self, clip_limit: float = 3.0, title_grid_size: Tuple[int, int] = (8, 8)):
        img = cv2.imread(self.CACHED_IMAGE, cv2.IMREAD_UNCHANGED)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=title_grid_size)
        cl    = clahe.apply(l)
        limg  = cv2.merge((cl, a, b))
        res   = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        ImagePIL.fromarray(res).save(self.CACHED_IMAGE)
        return self

    def reshape(self, width: int = 1000, height: int = 1000):
        img = cv2.imread(self.CACHED_IMAGE, cv2.IMREAD_UNCHANGED)
        res = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        ImagePIL.fromarray(res).save(self.CACHED_IMAGE)
        return self

    def to_segment(self, thresh: int = 0, maxval: int = 255):
        img  = cv2.imread(self.CACHED_IMAGE, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ImagePIL.fromarray(thresh).save(self.CACHED_IMAGE)
        return self

    def to_grey(self):
        ImagePIL.open(self.CACHED_IMAGE).convert('LA').save(self.CACHED_IMAGE)
        return self

    def show(self) -> PngImagePlugin.PngImageFile:
        return ImagePIL.open(self.CACHED_IMAGE)

    def toarray(self) -> np.ndarray:
        return np.asarray(self.show())

    def __del__(self):
        os.remove(self.CACHED_IMAGE)


class Tesseract:
    def __init__(self, oem: int = 3, psm: int = 3, lang: str = 'eng'):
        self.oem: int = oem
        self.psm: int = psm
        self.lang: str = lang
        self.lang_pool: Tuple[str, ...] = ('eng',)

    @staticmethod
    def unsharp(image: np.ndarray) -> np.ndarray:
        gaussian_3 = cv2.GaussianBlur(image, (9, 9), 10.0)
        return cv2.addWeighted(image, 1.4, gaussian_3, -0.5, 0, image)

    @staticmethod
    def median(image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, ksize=3)

    @staticmethod
    def resize(image: np.ndarray) -> np.ndarray:
        h, w, c = image.shape
        if h < 30:
            return cv2.resize(image, None, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        else:
            return image

    @staticmethod
    def threshold(image: np.ndarray) -> np.ndarray:
        median = int(np.mean(image))
        ret, thresh = cv2.threshold(image, median, 255, cv2.THRESH_BINARY_INV)
        return thresh

    @staticmethod
    def preprocess(image: np.ndarray) -> ImagePIL:
        return ImagePIL.fromarray(
                Tesseract.threshold(
                    cv2.cvtColor(
                        Tesseract.median(
                            Tesseract.unsharp(
                                Tesseract.resize(
                                    image
                                )
                            )
                        ),  cv2.COLOR_RGB2GRAY)
                )
        )


if __name__ == '__main__':
    image_path = os.path.join(DATA_DIR, 'images/', '84u2i4.jpg')
    prep = Preprocessor(image=np.asarray(ImagePIL.open(image_path)))
    res = prep.reshape().to_grey().to_segment().show()
    print(res)

    prep = Preprocessor(image=np.asarray(ImagePIL.open(image_path)))
    res = prep.to_segment().toarray()
    print(res)

    # new_image = Tesseract.preprocess(np.asarray(Image.open(path__to_crop_image)))
