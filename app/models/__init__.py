import os
import shutil

from models.craft.test import run_craft
from models.cropping import read_coord, run_to_crop

from models.pre_processor import Preprocessor
from models.post_processor import PostProcessor
from models.text_extractor import get_text
from models.deepTextRecognitionBenchmark.demo import TextRecognition

from config import DATA_DIR, APP_DIR, GPU

WEIGHT_FILE  = os.path.join(APP_DIR, 'models/detect/craft/craft_mlt_25k.pth')
DETECT_CACHE = os.path.join(APP_DIR, 'models/detect_cache')


text_recognizer = TextRecognition(
    saved_model       = os.path.join(APP_DIR, 'TPS-ResNet-BiLSTM-Attn.pth'),
    rgb               = False,
    PAD               = False,
    Transformation    = 'TPS',
    FeatureExtraction = 'ResNet',
    SequenceModeling  = 'BiLSTM',
    Prediction        = 'Attn',
    num_fiducial      = 20,
    input_channel     = 1,
    output_channel    = 512,
    hidden_size       = 256,
    workers           = 4,
    batch_size        = 192,
    batch_max_length  = 25,
    imgH              = 32,
    imgW              = 100,
    character         = '0123456789abcdefghijklmnopqrstuvwxyz',
    sensitive         = False,
    gpu               = GPU,
)

__all__ = [
    'Preprocessor',
    'PostProcessor',
    'get_text',
    'text_recognizer',
]


def detect_text(path_to_img: str) -> str:
    if os.path.exists(DETECT_CACHE):
        shutil.rmtree(DETECT_CACHE)
    os.mkdir(DETECT_CACHE)

    coord = run_craft(trained_model=WEIGHT_FILE, path_to_img=path_to_img)

    coordinates = read_coord(coord)
    for index in range(len(coordinates)):
        run_to_crop(path_to_img, coordinates[index], f'{DETECT_CACHE}/crop{index+1}.jpg')
    if not len(os.listdir(DETECT_CACHE)):
        return ''
    res = text_recognizer.recognize([os.path.join(DATA_DIR, 'temp/', i) for i in os.listdir(DETECT_CACHE)])
    return res.pred.str.join(' ')


if __name__ == '__main__':
    # from models import text_recognizer
    # res = text_recognizer.recognize([os.path.join(DATA_DIR, 'temp/', i) for i in os.listdir(os.path.join(DATA_DIR, 'temp/'))])
    # print(res)

    res = detect_text(path_to_img=os.path.join(DATA_DIR, 'images/0d9BSLO.jpg'))
    print(res)
