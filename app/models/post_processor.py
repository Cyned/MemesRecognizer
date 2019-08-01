import re

import numpy as np

from textblob import TextBlob
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from typing import Sequence


class PostProcessor(object):

    def __init__(self):
        self.spell = SpellChecker()

    def transform(self, samples: Sequence[str]) -> np.array:
        return np.array([self.correct(sample) for sample in tqdm(samples, total=len(samples), desc='Post processing')])

    def correct(self, text: str) -> str:
        text = self.remove_spec(text=text)
        text = self.text_blob(text=text)
        text = self.word_correction(text=text)
        return text

    @staticmethod
    def remove_spec(text: str) -> str:
        """ Remove special symbols """
        pattern = r"""[\["#\$%&\\'\(\)\*\+\-/<=>@\[\]\^_`\{|\}~]"""
        no_symb = re.sub(pattern, '', text.replace('\\n', '\n').replace('\n', ' '))
        return re.sub(r'[!.,?]+', '.', no_symb)

    def word_correction(self, text: str) -> str:
        a = word_tokenize(text=text)
        misspelled = self.spell.unknown(a)
        for word in misspelled:
            corrected = self.spell.correction(word)
            if corrected != word:
                text = re.sub(word, self.spell.correction(word), text, flags=re.IGNORECASE)
                continue

            for char in range(3, len(word)-3):
                if not self.spell.unknown([word[:char], word[char:]]):
                    text = re.sub(word, f'{word[:char]} {word[char:]}', text, flags=re.IGNORECASE)
                    break
        return text

    @staticmethod
    def text_blob(text: str) -> str:
        """ Correct a sequence """
        return str(TextBlob(text=text).correct())


if __name__ == '__main__':
    text = 'Hello, mu name is,!^ Ginga, but I donot understond English. Waitafuck...'
    print(f'Original:\t\t{text}')
    post = PostProcessor()
    res = post.correct(text=text)
    print('-' * 100)
    print(res)
