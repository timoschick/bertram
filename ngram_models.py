# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the logic for the form-only part of BERTRAM."""
import itertools
import random
from typing import List
from collections import Counter

import torch
import numpy as np
from torch import nn

import log
from utils import length_to_mask

logger = log.get_logger("root")

START_SYMBOL = '<S>'
END_SYMBOL = '</S>'
UNK_TOKEN = 'UNK'
PAD_TOKEN = 'PAD'

UNK_ID = 0
PAD_ID = 1


class NGramFeatures:
    def __init__(self, ngrams: List[str], ngram_ids: List[int]):
        self.ngrams = ngrams
        self.ngram_ids = ngram_ids
        self.ngram_lengths = len(ngram_ids)

    def __repr__(self):
        return '{}, {}, {}'.format(self.ngrams, self.ngram_ids, self.ngram_lengths)


class BatchedNGramFeatures:
    def __init__(self, ngram_ids: torch.Tensor, ngram_lengths: torch.Tensor):
        self.ngram_ids = ngram_ids
        self.ngram_lengths = ngram_lengths

    def __repr__(self):
        return '{}, {}'.format(self.ngram_ids, self.ngram_lengths)


class NGramBuilder:
    def __init__(self, vocab_file: str, ngram_threshold: int = 4, nmin: int = 3, nmax: int = 5, seed: int = None):

        self.nmin = nmin
        self.nmax = nmax

        self.ngram2id = {UNK_TOKEN: UNK_ID, PAD_TOKEN: PAD_ID}
        self.id2ngram = [UNK_TOKEN, PAD_TOKEN]

        ngram_counts = Counter()
        if seed is not None:
            random.seed(seed)

        with open(vocab_file, 'r', encoding='utf8') as file:

            for line in file:
                word = line.split()[0]
                ngram_counts.update(self.to_n_gram(word, self.nmin, self.nmax))

        most_common = ngram_counts.most_common()
        if self.nmin == self.nmax == 1:
            most_common = list(most_common)
            most_common.sort(key=lambda x: (-x[1], x[0]))

        for (ngram, count) in most_common:
            if count >= ngram_threshold:
                if ngram in self.ngram2id:
                    continue
                id_ = len(self.id2ngram)
                self.ngram2id[ngram] = id_
                self.id2ngram.append(ngram)

        logger.info('Found {} ngrams with min count {} and (nmin,nmax)=({},{}), first 10: {}, last 10: {}'.format(
            len(self.id2ngram), ngram_threshold, nmin, nmax, self.id2ngram[:10], self.id2ngram[-10:]
        ))

    def get_ngram_features(self, word: str, dropout_probability: float = 0) -> NGramFeatures:
        ngrams = self.to_n_gram(word, self.nmin, self.nmax, dropout_probability)
        ngram_ids = [self.ngram2id[ngram] if ngram in self.ngram2id else UNK_ID for ngram in ngrams]
        return NGramFeatures(ngrams, ngram_ids)

    @staticmethod
    def batchify(features: List[NGramFeatures]) -> BatchedNGramFeatures:
        ngram_ids = torch.tensor(np.array(
            list(itertools.zip_longest(*[x.ngram_ids for x in features], fillvalue=PAD_ID)),
            dtype=np.int32).T, dtype=torch.long)
        ngram_lengths = torch.tensor(np.array([x.ngram_lengths for x in features], dtype=np.int32), dtype=torch.long)
        return BatchedNGramFeatures(ngram_ids, ngram_lengths)

    @staticmethod
    def to_n_gram(word: str, nmin: int, nmax: int, dropout_probability: float = 0) -> List[str]:
        """
        Turns a word into a list of n-grams.
        :param word: the word
        :param nmin: the minimum number of characters per n-gram
        :param nmax: the maximum number of characters per n-gram
        :param dropout_probability: the probability of randomly removing an n-gram
        :return: the list of n-grams
        """
        ngrams = []

        if nmin == nmax:
            letters = [START_SYMBOL] + list(word) + [END_SYMBOL] + ([PAD_TOKEN] * max(10, (50 - len(list(word)))))
        else:
            letters = [START_SYMBOL] + list(word) + [START_SYMBOL]

        for i in range(len(letters)):
            for j in range(i + nmin, min(len(letters) + 1, i + nmax + 1)):
                ngram = ''.join(letters[i:j])
                ngrams.append(ngram)

        if dropout_probability > 0:
            ngrams = [ngram for ngram in ngrams if random.random() < (1 - dropout_probability)]

        if not ngrams:
            ngrams = [UNK_TOKEN]
        return ngrams

    def get_number_of_ngrams(self) -> int:
        return len(self.id2ngram)


class BagOfNgrams(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(BagOfNgrams, self).__init__()
        self.ngram_embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, ngram_ids, ngram_lengths):
        """
        :param ngram_ids:       shape is [batch_size x max_seq_length]
        :param ngram_lengths:   shape is [batch_size]
        """
        # shape is [batch_size x max_seq_length x embedding_size]
        ngrams_embedded = self.ngram_embeddings(ngram_ids)

        # shape is [batch_size x max_seq_length]
        mask = length_to_mask(ngram_lengths, max_len=ngram_ids.shape[1], dtype=torch.float)
        ngrams_embedded = ngrams_embedded * mask.unsqueeze(-1)

        bag_of_ngrams = torch.sum(ngrams_embedded, dim=1) / ngram_lengths.float().unsqueeze(-1)
        return bag_of_ngrams
