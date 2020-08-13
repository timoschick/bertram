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

"""This module contains the core BERTRAM architecture."""
import os

from typing import Callable, List, Dict
import jsonpickle
import torch
import torch.nn as nn
from torch.nn import MSELoss, Module, Embedding
from transformers import BertModel, BertConfig, RobertaModel, BertTokenizer, RobertaTokenizer, BertPreTrainedModel, \
    RobertaConfig, PreTrainedTokenizer, PreTrainedModel

import log
from input_processor import InputProcessor
from ngram_models import BagOfNgrams
from utils import length_to_mask

logger = log.get_logger("root")

IP_NAME = 'input_processor.json'
CONFIG_NAME = 'bertram_config.json'

MODE_FORM = 'form'
MODE_CONTEXT = 'context'
MODE_SHALLOW = 'shallow'
MODE_REPLACE = 'replace'
MODE_ADD = 'add'
MODE_ADD_QUOTES = 'add-quotes'

MODES = [MODE_FORM, MODE_CONTEXT, MODE_SHALLOW, MODE_REPLACE, MODE_ADD, MODE_ADD_QUOTES]


def requires_context(mode: str) -> bool:
    return mode != MODE_FORM


def requires_form(mode: str) -> bool:
    return mode != MODE_CONTEXT


def requires_sep(mode: str) -> bool:
    return mode in [MODE_ADD, MODE_ADD_QUOTES]


def requires_shallow(mode: str) -> bool:
    return mode == MODE_SHALLOW


class OverwriteableEmbedding(Module):
    """This Module is a wrapper around an Embedding Module, enabling embeddings for specific words to be overwritten."""

    def __init__(self, embedding: Embedding, overwrite_fct=None):
        super().__init__()
        self.embedding = embedding
        self.overwrite_fct = overwrite_fct

    def forward(self, inp: torch.Tensor):
        embds = self.embedding(inp)
        if self.overwrite_fct is not None:
            embds = self.overwrite_fct(embds)
        return embds


class BertramConfig:
    """This class contains the configuration for a BERTRAM instance."""

    def __init__(self, transformer_cls: str, output_size: int, mode: str, ngram_vocab_size: int):
        assert transformer_cls in MODELS.keys()
        assert mode in MODES

        self.transformer_cls = transformer_cls
        self.output_size = output_size
        self.mode = mode
        self.ngram_vocab_size = ngram_vocab_size

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf8') as f:
            f.write(jsonpickle.encode(self))

    @staticmethod
    def load(path: str) -> 'BertramConfig':
        with open(path, 'r', encoding='utf8') as f:
            cfg = jsonpickle.decode(f.read())
        return cfg


class ShallowCombination(nn.Module):
    """This Module can be used to generate a shallow combination from two embeddings using a gate."""

    def __init__(self, bertram_config: BertramConfig):
        super(ShallowCombination, self).__init__()
        self.linear = nn.Linear(2 * bertram_config.output_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.mode = bertram_config.mode

    def forward(self, embs1, embs2):
        embs_combined = torch.cat([embs1, embs2], dim=-1)
        a = self.sigmoid(self.linear(embs_combined))
        return a * embs1 + (1 - a) * embs2


class ReliabilityMeasure(nn.Module):
    """This Module implements an Attentive Mimicking head."""

    def __init__(self, config):
        super(ReliabilityMeasure, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, contexts, context_lengths):
        contexts = self.linear(contexts)
        mask = self._get_mask(context_lengths)
        similarities = self._get_context_similarities(contexts)
        similarities = self._mask_context_similarities(similarities, mask)
        reliabilities = self._get_context_reliabilities(similarities)
        return reliabilities

    @staticmethod
    def _get_mask(length):
        return length_to_mask(length, dtype=torch.float)

    @staticmethod
    def _get_context_reliabilities(similarities):
        reliabilities = torch.sum(similarities, dim=-1)
        reliability_sums = torch.sum(reliabilities, dim=1).unsqueeze(-1)
        reliabilities = reliabilities / (reliability_sums + 1e-12)
        return reliabilities

    @staticmethod
    def _mask_context_similarities(similarities, mask):
        mask_key = mask.unsqueeze(1)
        mask_query = mask.unsqueeze(2)
        return similarities * mask_key * mask_query

    @staticmethod
    def _get_context_similarities(contexts):
        """
        :param contexts: (batch_size x max_context_length x emb_dim)
        """
        Q = contexts
        K = torch.transpose(Q, dim0=-2, dim1=-1)
        d_k = torch.tensor(contexts.shape[-1], dtype=torch.float)
        similarities = torch.matmul(Q, K) / torch.sqrt(d_k)
        return similarities


class Bertram(BertPreTrainedModel):
    """This Module contains the core BERTRAM logic."""

    def __init__(self, transformer_config: BertConfig, bertram_config: BertramConfig, do_setup=False):

        super(Bertram, self).__init__(transformer_config)
        self.bertram_config = bertram_config
        self.transformer_config = transformer_config
        self.is_setup = False

        if requires_context(bertram_config.mode):
            transformer_cls, _, _ = MODELS[bertram_config.transformer_cls]
            setattr(self, bertram_config.transformer_cls, transformer_cls(transformer_config))

            self.reliability_measure = ReliabilityMeasure(transformer_config)
            self.linear = nn.Linear(transformer_config.hidden_size, bertram_config.output_size)
            self.init_weights()

        if requires_form(bertram_config.mode):
            self.ngram_processor = BagOfNgrams(bertram_config.ngram_vocab_size, bertram_config.output_size)

        if requires_shallow(bertram_config.mode):
            self.shallow_combination = ShallowCombination(self.bertram_config)

        if do_setup:
            self.setup()

    @property
    def transformer(self):
        """Get the underlying transformer language model (either a BERT instance or a RoBERTa instance)"""
        return getattr(self, self.bertram_config.transformer_cls)

    # noinspection PyUnresolvedReferences
    def setup(self):
        """Initialize the BERTRAM model and put a wrapper around the underlying transformer's embedding layer"""
        form_and_context = requires_context(self.bertram_config.mode) and requires_form(self.bertram_config.mode)
        if not isinstance(self.transformer.embeddings.word_embeddings, OverwriteableEmbedding) and form_and_context:
            word_embeddings = self.transformer.embeddings.word_embeddings
            self.transformer.embeddings.word_embeddings = OverwriteableEmbedding(word_embeddings)
        if requires_shallow(self.bertram_config.mode):
            if not hasattr(self, 'shallow_combination'):
                self.shallow_combination = ShallowCombination(self.bertram_config)
        self.is_setup = True

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                nr_of_contexts: torch.Tensor,
                mask_positions: torch.Tensor,
                attention_mask: torch.Tensor,
                ngram_ids: torch.Tensor,
                ngram_lengths: torch.Tensor,
                target_vectors: torch.Tensor = None):

        """
        Process a batch of words and contexts and generate embeddings. If `target_vectors` is not `None`,
        the loss is returned. Otherwise, the BERTRAM embeddings for all given words are returned.
        :param input_ids:           tensor of input token ids
        :param token_type_ids:      tensor of token type ids
        :param nr_of_contexts:      list of context lengths per word
        :param mask_positions:      tensor of shape sum(nr_of_contexts), containing the positions of the [MASK]
                                    tokens in the given contexts (assuming one per line)
        :param attention_mask:      attention mask tensor for the underlying transformer language model
        :param ngram_ids:           tensor of ngram ids for each word
        :param ngram_lengths:       list of ngram lengths (i.e., number of ngrams per word)
        :param target_vectors:      tensor containing the target vectors for each word (optional)
        """

        if not self.is_setup:
            raise ValueError("setup() must be called before using the model.")

        # if input has an additional 0th dimension with only one entry, it means we are in data parallel mode
        # and must first remove this additional dimension
        data_parallel_mode = input_ids is not None and len(input_ids.shape) == 3

        if data_parallel_mode:
            input_ids = torch.squeeze(input_ids, 0)
            token_type_ids = torch.squeeze(token_type_ids, 0)
            nr_of_contexts = torch.squeeze(nr_of_contexts, 0)
            mask_positions = torch.squeeze(mask_positions, 0)
            attention_mask = torch.squeeze(attention_mask, 0)
            ngram_ids = torch.squeeze(ngram_ids, 0)
            ngram_lengths = torch.squeeze(ngram_lengths, 0)
            target_vectors = torch.squeeze(target_vectors, 0)

        output_vectors = None
        ngram_vectors = None

        if requires_form(self.bertram_config.mode):
            ngram_vectors = self.ngram_processor(ngram_ids, ngram_lengths)
            if input_ids is None:
                return ngram_vectors

        if self.bertram_config.mode == MODE_FORM:
            output_vectors = ngram_vectors

        if requires_context(self.bertram_config.mode):
            overwrite_fct = None
            if self.bertram_config.mode == MODE_REPLACE:
                overwrite_fct = self.get_mask_oef(ngram_vectors, nr_of_contexts, mask_positions)
            elif requires_sep(self.bertram_config.mode):
                overwrite_fct = self.get_sep_oef(ngram_vectors, nr_of_contexts)

            self.transformer.embeddings.word_embeddings.overwrite_fct = overwrite_fct
            sequence_output, _ = self.transformer(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            self.transformer.embeddings.word_embeddings.overwrite_fct = None

            # get only the mask vector position for each sequence
            # shape = sum(nr_of_contexts) x emb_dim
            mask_output = self._get_mask_output(sequence_output, mask_positions)

            # regroup the sequence_output based on given lengths
            # shape = batch_size x max(nr_of_contexts) x emb_dim where batch_size := len(nr_of_contexts)
            grouped_mask_output = Bertram._group_sequence(mask_output, nr_of_contexts)

            # shape = batch_size x max(nr_of_contexts)
            reliability_scores = self.reliability_measure(grouped_mask_output, nr_of_contexts)

            output_vectors = self._get_weighted_sum(grouped_mask_output, reliability_scores)
            output_vectors = self.linear(output_vectors)

        if requires_shallow(self.bertram_config.mode):
            output_vectors = self.shallow_combination(output_vectors, ngram_vectors)

        if target_vectors is not None:
            loss_fct = MSELoss()
            loss = loss_fct(output_vectors, target_vectors)
            return loss
        else:
            return output_vectors

    @staticmethod
    def _get_weighted_sum(seq, weights):
        seq = seq * weights.unsqueeze(-1)
        wsum = torch.sum(seq, dim=1)
        return wsum

    @staticmethod
    def _get_mask_output(seq, indices):
        indices_one_hot = torch.zeros([seq.shape[0], seq.shape[1]]).to(seq.device).scatter_(1, indices.unsqueeze(-1), 1)
        seq_masked = seq * indices_one_hot.unsqueeze(-1)
        mask_output = torch.sum(seq_masked, dim=1)
        return mask_output

    @staticmethod
    def _group_sequence(seq, lengths):
        cum_len = 0
        y = []
        for idx, val in enumerate(lengths):
            y.append(seq[cum_len: cum_len + val])
            cum_len += val
        return torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

    @staticmethod
    def _duplicate(tensor_to_duplicate: torch.Tensor, nr_of_duplicates: torch.Tensor) -> List[torch.Tensor]:
        assert tensor_to_duplicate.shape[0] == nr_of_duplicates.shape[0]
        ret = []
        for ctx_idx, ctx_nr in enumerate(nr_of_duplicates):
            for _ in range(ctx_nr.item()):
                ret.append(tensor_to_duplicate[ctx_idx])
        return ret

    @staticmethod
    def get_mask_oef(ngram_vectors: torch.Tensor, nr_of_contexts: torch.Tensor, mask_positions: torch.Tensor) -> \
            Callable[[torch.Tensor], torch.Tensor]:
        """Generate a function for overwriting [MASK] token embeddings with given form-based ngram vectors."""

        ngram_vectors_duped = Bertram._duplicate(ngram_vectors, nr_of_contexts)

        def oef(embeddings: torch.Tensor):
            # embeddings has shape batch_size x max_context_length x emb_dim
            for batch_idx, mask_idx in enumerate(mask_positions):
                embeddings[batch_idx][mask_idx.item()] = ngram_vectors_duped[batch_idx]
            return embeddings

        return oef

    def get_sep_oef(self, ngram_vectors: torch.Tensor, nr_of_contexts: torch.Tensor) -> \
            Callable[[torch.Tensor], torch.Tensor]:
        """Generate a function for overwriting [SEP] token embeddings with given form-based ngram vectors."""

        ngram_vectors_duped = Bertram._duplicate(ngram_vectors, nr_of_contexts)
        # the ngram vector must always be injected at the same position:
        # [CLS] <NGRAM_VECTOR> : <CONTEXT> [SEP] for MODE_ADD
        # [CLS] " <NGRAM_VECTOR> " : <CONTEXT> [SEP] for MODE_ADD_QUOTES
        placeholder_idx = 2 if self.bertram_config.mode == MODE_ADD_QUOTES else 1

        def oef(embeddings: torch.Tensor):
            # embeddings has shape batch_size x max_context_length x emb_dim
            for batch_idx in range(embeddings.shape[0]):
                embeddings[batch_idx][placeholder_idx] = ngram_vectors_duped[batch_idx]
            return embeddings

        return oef


class BertramForRoberta(Bertram):
    """An instance of BERTRAM that used RoBERTa instead of BERT as the underlying language model"""
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, transformer_config: BertConfig, bertram_config: BertramConfig, do_setup=False):
        super(BertramForRoberta, self).__init__(transformer_config, bertram_config, do_setup)


MODELS = {
    'bert': (BertModel, BertTokenizer, Bertram),
    'roberta': (RobertaModel, RobertaTokenizer, BertramForRoberta)
}


class BertramWrapper:
    """
    This class is a wrapper for a trained BERTRAM model to allow for a straightforward combination
    with a pre-trained transformer model.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize a new wrapper from a given model directory
        :param model_path: the directory that contains the trained BERTRAM model
        :param device: the device to use for inferring word vectors
        """
        self.device = device

        # load the input processor corresponding to the model
        self.input_processor = InputProcessor.load(os.path.join(model_path, IP_NAME))
        self.input_processor.ngram_dropout = 0

        # load the model config and the actual model
        bertram_config = BertramConfig.load(os.path.join(model_path, CONFIG_NAME))
        _, _, bertram_cls = MODELS[bertram_config.transformer_cls]
        self.model, loading_info = bertram_cls.from_pretrained(model_path, bertram_config=bertram_config,
                                                               output_loading_info=True)  # type: Bertram

        if loading_info['missing_keys']:
            logger.info('Reloading with do_setup=True because of missing keys: {}'.format(loading_info))
            del self.model
            self.model, loading_info = bertram_cls.from_pretrained(model_path, bertram_config=bertram_config,
                                                                   output_loading_info=True,
                                                                   do_setup=True)  # type: Bertram
            if loading_info['missing_keys']:
                raise ValueError('Something went wrong loading a pretrained model: {}'.format(loading_info))

        self.model.setup()
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def infer_vector(self, word: str, contexts: List[str]) -> torch.Tensor:
        """
        Infer a word vector for a given word from its surface-form and a list of contexts
        :param word: the word
        :param contexts: the list of contexts; each context must contain `word` at least once.
        :return: the BERTRAM vector for the word
        """
        if self.model.bertram_config.mode == MODE_FORM:
            contexts = [word]

        if not contexts and self.model.bertram_config.mode == MODE_CONTEXT:
            raise ValueError("A context-only model cannot infer vectors without contexts.")

        batch = self.input_processor.generate_batch_from_input(word, contexts)
        return self.model(
            batch.input_ids.to(self.device) if contexts else None,
            batch.segment_ids.to(self.device),
            batch.nrs_of_contexts.to(self.device),
            batch.mask_positions.to(self.device),
            batch.input_mask.to(self.device),
            batch.ngram_features.ngram_ids.to(self.device),
            batch.ngram_features.ngram_lengths.to(self.device),
            None
        )[0].detach()

    def add_word_vectors_to_model(self, words_with_contexts: Dict[str, List[str]], tokenizer: PreTrainedTokenizer,
                                  model: PreTrainedModel) -> None:
        """
        Infer vectors for words and add them to the embedding matrix of a pre-trained transformer model. For each word
        `w` in `words_with_context.keys()`, a new token `<BERTRAM:w>` is added to the tokenizer's vocabulary and the
        corresponding BERTRAM vector is added to the model's embedding matrix. The token `<BERTRAM:w>` can then be used
        instead of (or in addition to) `w` like a regular token.
        :param words_with_contexts: a dictionary mapping words to lists of contexts in which they occur
        :param tokenizer: the transformer's tokenizer
        :param model: the transformer model
        """

        # infer embeddings for all words from their surface form and contexts
        embeddings = {word: self.infer_vector(word, contexts) for word, contexts in words_with_contexts.items()}

        # register the new words as special tokens in the tokenizer
        special_tokens = [f"<BERTRAM:{word}>" for word in embeddings.keys()]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        embedding_weight = _get_embeddings_module(model).word_embeddings.weight
        max_id = max(tokenizer.additional_special_tokens_ids)

        # if necessary, extend the transformer's embedding matrix to account for the new word vectors
        if embedding_weight.shape[0] <= max_id:
            filler = torch.zeros(max_id + 1 - embedding_weight.shape[0], embedding_weight.shape[1])
            new_embd = torch.nn.Parameter(torch.cat([embedding_weight, filler]).detach(), requires_grad=True)
            _get_embeddings_module(model).word_embeddings.weight = new_embd

        # add the word vectors for all words to the model's embedding matrix
        for word, embedding in embeddings.items():
            word_id = tokenizer.convert_tokens_to_ids(f"<BERTRAM:{word}>")
            _get_embeddings_module(model).word_embeddings.weight[word_id] = embedding


def _get_embeddings_module(model: PreTrainedModel):
    if hasattr(model, 'bert'):
        return model.bert.embeddings
    elif hasattr(model, 'roberta'):
        return model.roberta.embeddings
    else:
        return model.embeddings
