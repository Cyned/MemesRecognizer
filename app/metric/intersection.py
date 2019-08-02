import re

import pandas as pd
import numpy as np

from jiwer import wer as wer_
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts
from typing import Sequence, Callable, List
from os.path import join as path_join
from nltk.tokenize import word_tokenize

from config import DATA_DIR


def intersection(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> float:
    """
    Count the quality of the detection by its intersection
    :param y_true: List of [labeled words]
    :param y_pred: List of [detected words]
    :return: metric value
    """
    y_true = [{word.lower() for word in sample} for sample in y_true]
    y_pred = [{word.lower() for word in sample} for sample in y_pred]
    return np.mean([(len(true & pred) + 1e-5) / (len(true | pred) + 1e-5)
                    for true, pred in tqdm(zip(y_true, y_pred), total=len(y_true), desc='Count intersections')],
                   axis=0)


def wer(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """
    Count the quality of the detection by wer function
    :param y_true: List of strings
    :param y_pred: List of strings
    :return: metric value
    """
    return np.mean([wer_(truth=true, hypothesis=pred)
                    for true, pred in tqdm(zip(y_true, y_pred), total=len(y_true), desc='Count wer')],
                   axis=0)


def intersection_metric() -> Callable:
    """ Returns function to count the intersection metric """
    def closure(y_pred: List[str]) -> float:
        """
        Count the metric
        :param y_pred: List of [detected sequence]
        :return: metric value
        """
        y_pred = [text_detection_tokenizer(sample) for sample in y_pred]
        return intersection(y_true=y_test, y_pred=y_pred)

    data = pd.read_csv(path_join(DATA_DIR, 'eng_images.csv'), na_values='')
    data['caption'] = data.caption.fillna('')
    _, y_test = train_test_split(data['caption'])
    y_test = [text_detection_tokenizer(text=sample) for sample in y_test]
    return closure


def wer_metric() -> Callable:
    """ Returns function to count the wer metric """
    def closure(y_pred: List[str]) -> float:
        """
        Count the metric
        :param y_pred: List of [detected sequence]
        :return: metric value
        """
        y_pred = [' '.join(text_detection_tokenizer(sample)) for sample in y_pred]
        return wer(y_true=y_test, y_pred=y_pred)

    data = pd.read_csv(path_join(DATA_DIR, 'eng_images.csv'), na_values='')
    data['caption'] = data.caption.fillna('')
    _, y_test = train_test_split(data['caption'])
    y_test = [' '.join(text_detection_tokenizer(text=sample)) for sample in y_test]
    return closure


def train_test_split(data: pd.DataFrame):
    """
    Split data into two parts: train and test
    :param data: data set
    :return: train and test partitions
    """
    return tts(data, random_state=1341, shuffle=True, test_size=0.1)


def text_detection_tokenizer(text: str) -> List[str]:
    """
    Replace special symbols in the text (see pattern) and tokenize it
    :param text: text to tokenize
    :return: list of units/words
    """
    pattern = r"""[\["#\$%&\\'\(\)\*\+-/<=>@\[\]\^_`\{|\}~]"""
    return word_tokenize(re.sub(pattern, '', text.replace('\\n', '\n').replace('\n', ' ')))


if __name__ == '__main__':
    # TEST 1
    target = 'Hello i am data scientist'.split(' ')
    pred   = 'Hello Y am Data scientist'.split(' ')
    result = intersection(y_true=[target], y_pred=[pred])
    print(f'Metric: {result*100:.2f}%')

    # TEST 2
    data = pd.read_csv(path_join(DATA_DIR, 'eng_images.csv'))
    my_metric = intersection_metric()
    train, test = train_test_split(data=data)
    result = my_metric(y_pred=['hello world' for sample in range(test.shape[0])])
    print(f'Metric: {result*100:.2f}%')

    print(word_tokenize('Hello   words'))
