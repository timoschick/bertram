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

"""This file contains various classes and functions for preprocessing input before passing it to BERTRAM."""
from abc import ABC, abstractmethod
from shutil import copyfile

from transformers import PreTrainedTokenizer, GPT2Tokenizer
from typing import Dict, List, Optional
from collections import deque

import numpy as np
import random
import re
import jsonpickle
import torch

import bertram
import log
import utils

from ngram_models import NGramBuilder, NGramFeatures

logger = log.get_logger('root')

VOCAB_FILE_SUFFIX = '.vocab'


class RawInput:
    """Raw input for BERTRAM, consisting of a word and a list of corresponding contexts."""

    def __init__(self, word: str, contexts: List[str]):
        self.word = word
        self.contexts = contexts


class ProcessedInput:
    """
    Preprocessed input for BERTRAM, consisting of various tensors required for the underlying transformer,
    features for the ngram-model and (optionally) target vectors to be mimicked.
    """

    def __init__(self, input_ids: torch.Tensor, input_mask: torch.Tensor, segment_ids: torch.Tensor,
                 mask_positions: torch.Tensor, ngram_features: NGramFeatures, target_vector: Optional[torch.Tensor]):
        """
        :param input_ids: shape is [nr_of_contexts x max_seq_length]
        :param input_mask: shape is [nr_of_contexts x max_seq_length]
        :param segment_ids: shape is [nr_of_contexts x max_seq_length]
        :param mask_positions: shape is [nr_of_contexts]
        :param target_vector: shape is [emb_dim]
        """
        nr_of_contexts = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        assert list(input_ids.shape) == [nr_of_contexts, max_seq_length]
        assert list(input_mask.shape) == [nr_of_contexts, max_seq_length]
        assert list(segment_ids.shape) == [nr_of_contexts, max_seq_length]
        assert list(mask_positions.shape) == [nr_of_contexts]
        if target_vector is not None:
            assert len(list(target_vector.shape)) == 1

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_positions = mask_positions
        self.ngram_features = ngram_features
        self.target_vector = target_vector


class BatchedProcessedInputs:
    """This class represents a (zero-padded) batch of `ProcessedInput`s"""

    # noinspection PyUnresolvedReferences,PyCallingNonCallable
    def __init__(self, inputs: List[ProcessedInput]):
        self.input_ids = torch.cat([inp.input_ids for inp in inputs])
        self.input_mask = torch.cat([inp.input_mask for inp in inputs])
        self.segment_ids = torch.cat([inp.segment_ids for inp in inputs])
        self.nrs_of_contexts = torch.tensor([inp.input_ids.shape[0] for inp in inputs])
        self.mask_positions = torch.cat([inp.mask_positions for inp in inputs])
        self.ngram_features = NGramBuilder.batchify([inp.ngram_features for inp in inputs])

        batch_size = self.nrs_of_contexts.shape[0]
        nr_of_contexts = sum(self.nrs_of_contexts)
        max_seq_length = self.input_ids.shape[1]
        max_ngram_length = max(list(self.ngram_features.ngram_lengths))

        assert list(self.input_ids.shape) == [nr_of_contexts, max_seq_length]
        assert list(self.input_mask.shape) == [nr_of_contexts, max_seq_length]
        assert list(self.segment_ids.shape) == [nr_of_contexts, max_seq_length]
        assert list(self.mask_positions.shape) == [nr_of_contexts]
        assert list(self.nrs_of_contexts.shape) == [batch_size]
        assert list(self.ngram_features.ngram_lengths.shape) == [batch_size]
        assert list(self.ngram_features.ngram_ids.shape) == [batch_size, max_ngram_length]

        if all([inp.target_vector is None for inp in inputs]):
            self.target_vectors = None
        else:
            self.target_vectors = torch.stack([inp.target_vector for inp in inputs])
            assert len(list(self.target_vectors.shape)) == 2 and list(self.target_vectors.shape)[0] == batch_size

    def __repr__(self):
        return '{} , {}, {}, {}'.format(self.input_ids.shape, self.nrs_of_contexts.shape, self.mask_positions.shape,
                                        self.target_vectors.shape)


class EndOfDatasetException(Exception):
    pass


class InputBuffer:
    """This class represents an input buffer that stores batches of input to BERTRAM during training."""

    def __init__(self, input_processor: 'InputProcessor'):
        self.input_processor = input_processor
        self.words = deque()  # type: deque[str]
        self.contexts = {}  # type: Dict[str, deque[str]]

    def __repr__(self):
        return '{} ({})'.format(self.words, self.contexts)

    def fill_from_file(self, file_path: str, shuffle_words=True, form_only=False):
        """
        Fill the input buffer from a file containing training instances. Each line of this file should be
        of the form <WORD><TAB><CONTEXT_1><TAB>...<CONTEXT_n>
        """
        if form_only:
            for word in self.input_processor.word_counts:
                occurrences = self.input_processor.get_occurrences(word)
                if occurrences > 0:
                    self.words.extend([word] * occurrences)
                    self.contexts[word] = deque([word] * occurrences * len(self.input_processor.train_files))
        else:
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    self._fill_from_line(line, form_only=form_only)

        if shuffle_words:
            self._shuffle()

    def _fill_from_line(self, line: str, shuffle_contexts=True, form_only=False):

        if not form_only:
            comps = re.split(r'\t', line)
            word = comps[0]
            all_contexts = [x for x in comps[1:] if word in x.split()]
            if shuffle_contexts:
                random.shuffle(all_contexts)

            if not all_contexts:
                return

            word_occurrences = self.input_processor.get_occurrences(word, len(all_contexts))

        else:
            word, _ = line.split('\t', 1)
            word_occurrences = self.input_processor.get_occurrences(word)
            all_contexts = [word] * (2 * word_occurrences)

        if word_occurrences > 0:
            self.words.extend([word] * word_occurrences)
            self.contexts[word] = deque(all_contexts)

    def _shuffle(self):
        word_list = list(self.words)
        random.shuffle(word_list)
        self.words = deque(word_list)

    def has_next(self) -> bool:
        return len(self.words) > 0

    def length(self) -> int:
        return len(self.words)

    def get(self, nr_of_contexts: int) -> RawInput:
        word = self.words.pop()
        contexts = []

        for _ in range(nr_of_contexts):
            if not self.contexts[word]:
                logger.warning('Ran out of contexts for word "{}"'.format(word))
                break
            contexts.append(self.contexts[word].pop())

        if not contexts:
            return self.get(nr_of_contexts)
        else:
            return RawInput(word, contexts)

    def reset(self):
        self.words.clear()
        self.contexts.clear()


class AbstractInputProcessor(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def generate_batch_from_buffer(self, batch_size: int) -> BatchedProcessedInputs:
        """Generate a new batch of inputs for training from the input processor's underlying buffer"""
        pass

    @abstractmethod
    def generate_batch_from_input(self, word: str, contexts: List[str]) -> BatchedProcessedInputs:
        """Generate a new batch of inputs from the given word and its set of contexts"""
        pass

    @abstractmethod
    def get_number_of_train_examples_per_epoch(self) -> int:
        pass


class InputProcessor(AbstractInputProcessor):
    """
    This class provides processed inputs for BERTRAM both during training and testing.
    """

    def __init__(self, word_embeddings_file: str, word_embeddings_format: str, train_files: List[str], vocab_file: str,
                 vector_size: int, ngram_threshold: int = 4, nmin: int = 3, nmax: int = 5, ngram_dropout: float = 0,
                 min_word_count: int = 100, max_copies: int = 5, smin: int = 20, smax: int = 20,
                 max_seq_length: int = 128, model_cls='bert', bert_model='bert-base-uncased',
                 mode: Optional[str] = None, seed: int = None, form_only: bool = False, sep_symbol: str = ':'):
        """
        Initialize a new input processor.
        :param word_embeddings_file: the file containing all target word embeddings for training.
        :param word_embeddings_format: the format in which the embeddings are stored (either 'text' or 'gensim').
        :param train_files: the training files, containing lines of the form <WORD><TAB><CONTEXT_1><TAB>...<CONTEXT_n>
        :param vocab_file: the vocabulary to be used for training
        :param vector_size: the size of the target vectors
        :param ngram_threshold: the minimum number of occurences for a ngram to get its own embedding
        :param nmin: the minimum length (in characters) for an ngram to get its own embedding
        :param nmax: the maximum length (in characters) for an ngram to get its own embedding
        :param ngram_dropout: the probability that a ngram is randomly removed during training
        :param min_word_count: the minimum number of contexts for a word to be used as a training instance
        :param max_copies: the maximum number of copies of a word in a single training epoch
        :param smin: the minimum number of contexts per word
        :param smax: the maximum number of contexts per word
        :param max_seq_length: the maximum sequence length (in tokens) to be considered
        :param model_cls: the underlying transformer model's class (either RobertaModel or BertModel)
        :param bert_model: the BERT model to be used (e.g., bert-base-uncased, roberta-large, ...)
        :param mode: the mode of the BERTRAM model for which this input processor is used
        :param seed: the seed to be used for RNG initialization
        :param form_only: whether this input processor should generate inputs for a form-only model
        :param sep_symbol: the separation symbol to be used for the BERTRAM-ADD configuration
        """

        if seed:
            random.seed(seed)

        self.model_cls = model_cls
        self.bert_model = bert_model
        self.word_embeddings_file = word_embeddings_file
        self.word_embeddings_format = word_embeddings_format
        self.train_files = train_files
        self.vocab_file = vocab_file

        self.vector_size = vector_size
        self.ngram_threshold = ngram_threshold
        self.nmin = nmin
        self.nmax = nmax
        self.ngram_dropout = ngram_dropout
        self.min_word_count = min_word_count
        self.max_copies = max_copies
        self.smin = smin
        self.smax = smax
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.form_only = form_only
        self.sep_symbol = sep_symbol

        self.tokenizer = None  # type: Optional[PreTrainedTokenizer]
        self.word_counts = None  # type: Optional[Dict[str,int]]
        self.word_embeddings = None  # type: Optional[Dict[str, np.ndarray]]
        self.ngram_builder = None  # type: Optional[NGramBuilder]
        self.buffer = None  # type: Optional[InputBuffer]

        self.setup()
        self.train_file_idx = 0
        self.reset()

    def setup(self):
        """Instantiate the given input processor and fill its underlying input buffer with examples"""
        self.buffer = InputBuffer(self)
        _, tokenizer_cls, _ = bertram.MODELS[self.model_cls]
        self.tokenizer = tokenizer_cls.from_pretrained(self.bert_model)

        self.ngram_builder = NGramBuilder(self.vocab_file, self.ngram_threshold, self.nmin, self.nmax)
        self.word_counts = {}

        with open(self.vocab_file, 'r', encoding='utf8') as file:
            for line in file:
                word, count = line.split()
                self.word_counts[word] = int(count)

        if hasattr(self, 'word_embeddings_file') and self.word_embeddings_file is not None:
            self.word_embeddings = utils.load_embeddings(self.word_embeddings_file, self.word_embeddings_format)

    def reset(self) -> None:
        random.shuffle(self.train_files)
        self.buffer.reset()
        self.train_file_idx = 0

    def generate_batch_from_buffer(self, batch_size: int, num_chunks=-1) -> BatchedProcessedInputs:

        if self.buffer.length() < batch_size:
            self._fill_buffer()

        if self.buffer.length() < batch_size:
            raise EndOfDatasetException()

        logger.debug("Buffer is filled...")

        # divide batch_size into random multiples of 2
        nrs_of_contexts = utils.generate_chunks(batch_size, min_val=self.smin, max_val=self.smax, num_chunks=num_chunks)
        raw_inputs = []  # type: List[RawInput]

        logger.debug("Nrs of contexts: {}".format(nrs_of_contexts))

        for nr_of_contexts in nrs_of_contexts:
            raw_inputs.append(self.buffer.get(nr_of_contexts))

        logger.debug('Done creating raw inputs')

        processed_inputs = [self.generate_processed_input(raw_input, with_target=True) for raw_input in raw_inputs]
        logger.debug('Done creating processed inputs')
        return BatchedProcessedInputs(processed_inputs)

    def generate_batch_from_input(self, word: str, contexts: List[str]) -> BatchedProcessedInputs:
        raw_input = RawInput(word=word, contexts=contexts)
        processed_inputs = [self.generate_processed_input(raw_input, with_target=False)]
        return BatchedProcessedInputs(processed_inputs)

    def get_occurrences(self, word, count=-1):

        if word not in self.word_counts or word not in self.word_embeddings:
            return 0
        if count < 0:
            count = self.word_counts[word]

        if count < self.min_word_count:
            return 0
        return int(max(1, min(int(count / self.min_word_count), self.max_copies)))

    def _fill_buffer(self) -> bool:
        # select the next train file
        if self.train_file_idx == len(self.train_files):
            logger.info('Reached the end of the dataset')
            return False

        while self.train_file_idx < len(self.train_files):
            train_file = self.train_files[self.train_file_idx]
            logger.info('Processing training file {} of {}: {}'.format(self.train_file_idx + 1, len(self.train_files),
                                                                       train_file))
            self.train_file_idx += 1
            self.buffer.fill_from_file(train_file, form_only=self.form_only)
            logger.info('Done processing training file, batch size is {}'.format(self.buffer.length()))
        return True

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['word_counts']
        del odict['word_embeddings']
        del odict['tokenizer']
        del odict['buffer']
        del odict['ngram_builder']
        return odict

    def __setstate__(self, d):
        self.__dict__.update(d)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf8') as f:
            f.write(jsonpickle.encode(self))
        copyfile(self.vocab_file, path + VOCAB_FILE_SUFFIX)

    @classmethod
    def load(cls, path: str, setup=True) -> 'InputProcessor':
        with open(path, 'r', encoding='utf8') as f:
            batch_builder = jsonpickle.decode(f.read())  # type: InputProcessor
            batch_builder.vocab_file = path + VOCAB_FILE_SUFFIX

            if not hasattr(batch_builder, 'form_only'):
                batch_builder.form_only = False
            if not hasattr(batch_builder, 'sep_symbol'):
                batch_builder.sep_symbol = ':'

            if setup:
                batch_builder.setup()
        return batch_builder

    def _truncate(self, word_tokens: List[str], context_tokens: List[str]) -> None:

        if self.mode == bertram.MODE_ADD_QUOTES:
            max_len = self.max_seq_length - 7
        elif bertram.requires_sep(self.mode):
            max_len = self.max_seq_length - 5
        else:
            max_len = self.max_seq_length - 3

        while len(word_tokens) + len(context_tokens) > max_len:

            if len(word_tokens) > len(context_tokens):
                del word_tokens[-1]
            else:
                mask_idx = context_tokens.index(self.tokenizer.mask_token)
                if mask_idx > (len(context_tokens) - 1) / 2:
                    del context_tokens[0]
                else:
                    del context_tokens[-1]

    def _replace_word_with_mask(self, context: str, word: str) -> str:
        words = context.split()
        return ' '.join(self.tokenizer.mask_token if w == word else w for w in words)

    # noinspection PyCallingNonCallable
    def generate_processed_input(self, raw_input: RawInput, with_target: bool = False) -> ProcessedInput:

        if not raw_input.contexts:
            raw_input.contexts = [raw_input.word + ' .']

        all_input_ids, all_input_mask, all_segment_ids, all_mask_positions = [], [], [], []
        target_vector = torch.tensor(self.word_embeddings[raw_input.word], dtype=torch.float) if with_target else None

        ngram_features = self.ngram_builder.get_ngram_features(raw_input.word, self.ngram_dropout)

        for context in raw_input.contexts:

            word = raw_input.word
            context = self._replace_word_with_mask(context, word)

            if isinstance(self.tokenizer, GPT2Tokenizer):
                context_tokens = self.tokenizer.tokenize(context, add_prefix_space=True)
            else:
                context_tokens = self.tokenizer.tokenize(context)

            self._truncate([], context_tokens)

            if self.tokenizer.mask_token not in context_tokens:
                logger.warning('Skipping context "{}" (does not contain the target word "{}")'.format(context, word))
                continue

            prefix = [self.tokenizer.cls_token]
            if bertram.requires_sep(self.mode):
                if self.mode == bertram.MODE_ADD_QUOTES:
                    quote_symbol = 'Ġ"' if isinstance(self.tokenizer, GPT2Tokenizer) else '"'
                    prefix += [quote_symbol, self.tokenizer.sep_token, quote_symbol, self.sep_symbol]
                else:
                    prefix += [self.tokenizer.sep_token, self.sep_symbol]

            tokens = prefix + context_tokens + [self.tokenizer.sep_token]
            mask_position = tokens.index(self.tokenizer.mask_token)
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            all_mask_positions.append(mask_position)

        all_input_ids = torch.tensor(all_input_ids)
        all_input_mask = torch.tensor(all_input_mask)
        all_segment_ids = torch.tensor(all_segment_ids)
        all_mask_positions = torch.tensor(all_mask_positions)

        if len(all_input_ids.shape) != 2:
            logger.warning("Input Ids have shape {} for word '{}', contexts '{}'"
                           .format(all_input_ids.shape, raw_input.word, raw_input.contexts))

        return ProcessedInput(all_input_ids, all_input_mask, all_segment_ids, all_mask_positions, ngram_features,
                              target_vector)

    def get_number_of_train_examples_per_epoch(self):
        return sum([self.get_occurrences(word) for word in self.word_counts])
