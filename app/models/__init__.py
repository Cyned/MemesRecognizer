import os

from models.pre_processor import Preprocessor
from models.post_processor import PostProcessor
from models.text_extractor import get_text
from models.deepTextRecognitionBenchmark.demo import TextRecognition

from config import DATA_DIR, APP_DIR, GPU

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

if __name__ == '__main__':
    from models import text_recognizer
    res = text_recognizer.recognize([os.path.join(DATA_DIR, 'temp/', i) for i in os.listdir(os.path.join(DATA_DIR, 'temp/'))])
    print(res)
