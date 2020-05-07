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

"""
This module contains the script for training a new BERTRAM instance.
Parts of this file are copied from https://github.com/huggingface/transformers/blob/master/examples/run_glue.py
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import sys

import bertram
import log
import visdom

import numpy as np
import torch
from tqdm import trange

from transformers import AdamW, WarmupLinearSchedule, WEIGHTS_NAME, CONFIG_NAME as TRANSFORMER_CONFIG_NAME, \
    PYTORCH_TRANSFORMERS_CACHE

from input_processor import InputProcessor, EndOfDatasetException
from bertram import Bertram, IP_NAME, CONFIG_NAME

logger = log.get_logger('root')


def main(args):
    parser = argparse.ArgumentParser("Train a new BERTRAM instance")

    # Required parameters
    parser.add_argument('--model_cls', default='bert', choices=['bert', 'roberta'],
                        help="The transformer model class, either 'bert' or 'roberta'.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="The pretrained model to use (e.g., 'bert-base-uncased'.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--train_dir', type=str, required=True,
                        help="The directory in which the buckets for training are stored. "
                             "Each bucket should be a text file containing lines of the form "
                             "<WORD><TAB><CONTEXT_1><TAB>...<CONTEXT_n>")
    parser.add_argument('--vocab', type=str, required=True,
                        help="The file in which the vocabulary to be used for training is stored."
                             "Each line should be of the form <WORD> <COUNT>")
    parser.add_argument('--emb_file', type=str, required=True,
                        help="The file in which the target embeddings for mimicking are stored.")
    parser.add_argument('--emb_dim', type=int, required=True,
                        help="The number of dimensions for the target embeddings.")
    parser.add_argument('--mode', choices=bertram.MODES, required=True,
                        help="The BERTRAM mode (e.g., 'form', 'add', 'replace').")

    # Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--save_epochs', type=int, nargs='+', default=[1, 5, 10],
                        help="The number of epochs after which a checkpoint is saved.")
    parser.add_argument('--min_word_count', '-mwc', type=int, default=100,
                        help="The minimum number of occurrences for a word to be used as training target.")
    parser.add_argument('--num_buckets', type=int, default=25,
                        help="The number of buckets in the training directory.")
    parser.add_argument('--emb_format', type=str, choices=['text', 'gensim'], default='text',
                        help="The format in which target embeddings are stored.")
    parser.add_argument('--no_finetuning', action='store_true',
                        help="Whether not to finetune the underlying transformer language model.")
    parser.add_argument('--optimize_only_combinator', action='store_true',
                        help="Whether to freeze both the underyling transformer language model and the ngram"
                             "embeddings during training.")

    # Context parameters
    parser.add_argument('--smin', type=int, default=20,
                        help="The minimum number of contexts per word.")
    parser.add_argument('--smax', type=int, default=20,
                        help="The maximum number of contexts per word.")

    # Form parameters
    parser.add_argument('--nmin', type=int, default=3,
                        help="The minimum number of characters per ngram.")
    parser.add_argument('--nmax', type=int, default=5,
                        help="The maximum number of characters per ngram.")
    parser.add_argument('--dropout', type=float, default=0,
                        help="The ngram dropout probability.")
    parser.add_argument('--ngram_threshold', type=int, default=4,
                        help="The minimum number of occurrences for an ngram to get its own embedding.")

    # Visdom parameters
    parser.add_argument('--visdom_port', type=int, default=8098)
    parser.add_argument('--visdom_server', type=str, default=None)

    args = parser.parse_args(args)
    vis = visdom.Visdom(port=args.visdom_port, server=args.visdom_server) if args.visdom_server else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps: {} < 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # generate output directories for all save_epochs
    for epoch in range(int(args.num_train_epochs)):
        if (epoch + 1) in args.save_epochs:
            out_dir = args.output_dir + '-e' + str(epoch + 1)
            if os.path.exists(out_dir) and os.listdir(out_dir):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(out_dir))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

    train_files = [args.train_dir + 'train.bucket' + str(i) + '.txt' for i in range(args.num_buckets)]
    _, _, bertram_cls = bertram.MODELS[args.model_cls]

    if args.bert_model in ['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']:
        logger.info("Initializing new BERTRAM instance from {}.".format(args.bert_model))

        input_processor = InputProcessor(
            word_embeddings_file=args.emb_file,
            word_embeddings_format=args.emb_format,
            train_files=train_files,
            vocab_file=args.vocab,
            vector_size=args.emb_dim,
            nmin=args.nmin,
            nmax=args.nmax,
            ngram_dropout=args.dropout,
            ngram_threshold=args.ngram_threshold,
            smin=args.smin,
            smax=args.smax,
            min_word_count=args.min_word_count,
            max_seq_length=args.max_seq_length,
            form_only=(args.mode == bertram.MODE_FORM),
            model_cls=args.model_cls,
            bert_model=args.bert_model
        )

        cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_TRANSFORMERS_CACHE, 'distributed_-1')

        bertram_config = bertram.BertramConfig(
            transformer_cls=args.model_cls, output_size=args.emb_dim, mode=args.mode,
            ngram_vocab_size=input_processor.ngram_builder.get_number_of_ngrams()
        )

        model = bertram_cls.from_pretrained(args.bert_model, cache_dir=cache_dir, bertram_config=bertram_config)

    else:
        logger.info("Initializing pretrained BERTRAM instance from {}.".format(args.bert_model))

        input_processor = InputProcessor.load(os.path.join(args.bert_model, IP_NAME))
        bertram_config = bertram.BertramConfig.load(os.path.join(args.bert_model, CONFIG_NAME))
        model, loading_info = bertram_cls.from_pretrained(args.bert_model, bertram_config=bertram_config,
                                                          output_loading_info=True)  # type: Bertram
        if loading_info['missing_keys']:
            raise ValueError('Something went wrong loading a pretrained model: {}'.format(loading_info))

        if args.mode != model.bertram_config.mode:
            logger.warning("Overwriting original mode {} with {}.".format(model.bertram_config.mode, args.mode))
            model.bertram_config.mode = args.mode
            input_processor.mode = args.mode

    model.setup()
    model = model.to(device)

    model_orig = model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if args.no_finetuning:
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if ('bert.' not in n and 'roberta.' not in n)],
            'weight_decay': 0.01
        }]
        model_orig.transformer.eval()
        for name, param in model_orig.transformer.named_parameters():
            param.requires_grad = False

    elif args.optimize_only_combinator:
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if
                       ('bert.' not in n and 'roberta.' not in n) and 'ngram_processor.' not in n],
            'weight_decay': 0.01
        }]

        for name, param in model_orig.named_parameters():
            if 'bert.' in name or 'roberta.' in name or 'ngram_processor.' in name:
                param.requires_grad = False

    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_train_examples = input_processor.get_number_of_train_examples_per_epoch()

    avg_contexts_per_word = (args.smin + args.smax) / 2
    avg_examples_per_batch = args.train_batch_size / avg_contexts_per_word

    num_train_optimization_steps = int(
        num_train_examples / avg_examples_per_batch / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.mode == bertram.MODE_FORM:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                                     t_total=num_train_optimization_steps)
    global_step = 0

    if vis is not None:
        loss_window = vis.line(Y=np.array([0]), X=np.array([0]),
                               opts=dict(xlabel='Step', ylabel='Loss', title='Training loss (' + args.output_dir + ')',
                                         legend=['Loss']))
    else:
        loss_window = None

    model.train()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        summed_loss = 0
        step = 0

        while True:
            try:
                nr_of_parallel_batches = max(1, n_gpu)
                batches = []

                # fix a number of chunks for the current set of batches to make parallel processing feasible
                num_chunks = -1

                for i in range(nr_of_parallel_batches):
                    batch = input_processor.generate_batch_from_buffer(args.train_batch_size, num_chunks=num_chunks)
                    batches.append(batch)
                    num_chunks = len(batch.nrs_of_contexts)

                input_ids = torch.stack([batch.input_ids for batch in batches]).to(device)
                segment_ids = torch.stack([batch.segment_ids for batch in batches]).to(device)
                nrs_of_contexts = torch.stack([batch.nrs_of_contexts for batch in batches]).to(device)
                mask_positions = torch.stack([batch.mask_positions for batch in batches]).to(device)
                input_mask = torch.stack([batch.input_mask for batch in batches]).to(device)

                # pad ngram ids
                if len(batches) > 1:
                    max_nr_of_ngrams = max([batch.ngram_features.ngram_ids.shape[1] for batch in batches])
                    for batch in batches:
                        batch_nr_of_ngrams = batch.ngram_features.ngram_ids.shape[1]
                        nr_of_words = batch.ngram_features.ngram_ids.shape[0]
                        padding = torch.zeros((nr_of_words, max_nr_of_ngrams - batch_nr_of_ngrams), dtype=torch.long)
                        batch.ngram_features.ngram_ids = torch.cat([batch.ngram_features.ngram_ids, padding], dim=1)

                ngram_ids = torch.stack([batch.ngram_features.ngram_ids for batch in batches]).to(device)
                ngram_lengths = torch.stack([batch.ngram_features.ngram_lengths for batch in batches]).to(device)
                target_vectors = torch.stack([batch.target_vectors for batch in batches]).to(device)

                loss = model(input_ids, segment_ids, nrs_of_contexts, mask_positions, input_mask,
                             ngram_ids, ngram_lengths, target_vectors)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                summed_loss += loss.item()

                logger.debug('Done with backward step')

                if step > 0 and step % 100 == 0:

                    if vis is not None:
                        vis.line(Y=np.array([summed_loss / 100]), X=np.array([global_step]), win=loss_window,
                                 update='append')

                    logger.info('Step: %d\tLoss: %.17f', step, (summed_loss / 100))
                    summed_loss = 0

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                step += 1

            except EndOfDatasetException:
                logger.info('Done with epoch %d', epoch)
                input_processor.reset()

                if (epoch + 1) in args.save_epochs:
                    out_dir = args.output_dir + '-e' + str(epoch + 1)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(out_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(out_dir, TRANSFORMER_CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                    output_ip_file = os.path.join(out_dir, IP_NAME)
                    input_processor.save(output_ip_file)

                    output_bc_file = os.path.join(out_dir, CONFIG_NAME)
                    model_to_save.bertram_config.save(output_bc_file)

                break

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(args.output_dir, TRANSFORMER_CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    output_ip_file = os.path.join(args.output_dir, IP_NAME)
    input_processor.save(output_ip_file)

    output_bc_file = os.path.join(args.output_dir, CONFIG_NAME)
    model_to_save.bertram_config.save(output_bc_file)


if __name__ == "__main__":
    main(sys.argv[1:])
