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

"""This module contains various utility functions."""
import io
import random

import numpy as np
import torch
from gensim.models import Word2Vec

import log

logger = log.get_logger('root')


def load_embeddings(file_name: str, embeddings_format: str):
    """Load word embeddings from the given file, either in plain-text format ('text') or in gensim format ('gensim')"""
    if embeddings_format == 'text':
        return _load_w2v_model_file(filename=file_name)
    else:
        return _load_w2v_model_gensim(filename=file_name)


def _load_w2v_model_gensim(filename):
    """Load word embeddings from a given file in gensim format"""
    logger.info('Loading embeddings from %s', filename)
    w2v_model = Word2Vec.load(filename)
    word2vec = w2v_model.wv
    del w2v_model
    logger.info('Done loading embeddings')
    return word2vec


def _load_w2v_model_file(filename):
    """Load word embeddings from a given file in plain-text format"""
    logger.info('Loading embeddings from %s', filename)
    w2v = DummyDict()
    w2v.update({w: v for w, v in _load_vectors(filename)})
    w2v.vocab = w2v.keys()
    logger.info('Done loading embeddings')
    return w2v


def _load_vectors(path, skip=False):
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if skip:
                skip = False
            else:
                index = line.index(' ')
                word = line[:index]
                yield word, np.array([np.float(entry) for entry in line[index + 1:].split()])


class DummyDict(dict):
    pass


def generate_chunks(total, min_val, max_val, num_chunks=-1):
    """
    Randomly generate a list of integers l such that sum(l) = total and for each x in l, min_val <= x <= max_val.
    If num_chunks > 0, it is guaranteed that the list contains exactly num_chunks elements.
    """

    if num_chunks <= 0:
        chunks = []
        while total > 0:
            next_chunk_size = random.randint(min(total, min_val), min(total, max_val))
            if 0 < total - next_chunk_size < min_val:
                continue
            total -= next_chunk_size
            chunks.append(next_chunk_size)
        return chunks
    else:
        if total < num_chunks * min_val:
            raise ValueError('Total ({}) must be >= num_chunks * min_val ({}*{})'.format(total, num_chunks, min_val))

        if total > num_chunks * max_val:
            raise ValueError('Total ({}) must be <= num_chunks * max_val ({}*{})'.format(total, num_chunks, max_val))

        total -= num_chunks * min_val
        chunks = None
        while not chunks or any([x > max_val for x in chunks]):
            split_points = [0, total]
            for _ in range(num_chunks - 1):
                split_points.append(random.randint(0, total))
            split_points.sort()
            chunks = [split_points[i + 1] - split_points[i] + min_val for i in range(len(split_points) - 1)]

        return chunks


def length_to_mask(length, max_len=None, dtype=None):
    """
    Convert a 1d tensor of lengths to the corresponding mask tensor.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
