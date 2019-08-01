import os
import shutil

from models.craft.test import CraftModel
from models.cropping import read_coord, run_to_crop

from models.pre_processor import Preprocessor
from models.post_processor import PostProcessor
from models.text_extractor import get_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from models.deepTextRecognitionBenchmark.demo import TextRecognition
from timeit_ import timeit_context

from config import DATA_DIR, APP_DIR, GPU

WEIGHT_FILE  = os.path.join(APP_DIR, 'models/craft/craft_mlt_25k.pth')
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

craft_model = CraftModel(trained_model=WEIGHT_FILE)
post_processor = PostProcessor()
analyser = SentimentIntensityAnalyzer()


__all__ = [
    'Preprocessor',
    'PostProcessor',
    'get_text',
    'text_recognizer',
    'craft_model',
    'detect_text',
]


def detect_text(path_to_img: str, postprocess: bool = False, verbose: bool = False) -> str:
    """
    Detect text on image
    :param path_to_img: path to the image
    :param postprocess: if use post process methods to the output text
    :param verbose: printing the progress of pipeline
    :return: text from the image
    """
    if verbose:
        print('Preparing cache folder...')
    if os.path.exists(DETECT_CACHE):
        shutil.rmtree(DETECT_CACHE)
    os.mkdir(DETECT_CACHE)

    if verbose:
        print('Detecting text on image...')
    coord = craft_model.detect(path_to_img=path_to_img)

    if verbose:
        print('Rotating and croping the image...')
    coordinates = read_coord(coord)
    for index in range(len(coordinates)):
        if coordinates[index]:
            run_to_crop(path_to_img, coordinates[index], os.path.join(DETECT_CACHE, f'crop{index+1}.jpg'))
    if not len(os.listdir(DETECT_CACHE)):
        if verbose:
            print('There is no text on the image.')
        return ''
    if verbose:
        print('Recognizing text...')
    result = text_recognizer.recognize([os.path.join(DETECT_CACHE, i) for i in os.listdir(DETECT_CACHE)])
    text = ' '.join(result.pred)
    if postprocess:
        if verbose:
            print('Post processing the text...')
        return post_processor.correct(text)
    return text


if __name__ == '__main__':
    # from models import text_recognizer
    # res = text_recognizer.recognize(
    #     [os.path.join(DATA_DIR, 'temp/', i) for i in os.listdir(os.path.join(DATA_DIR, 'temp/'))],
    # )
    # print(res)
    with timeit_context('Text detect'):
        res = detect_text(path_to_img=os.path.join(DATA_DIR, 'images/0SLMQpP.jpg'), postprocess=False, verbose=True)
        print(res)
