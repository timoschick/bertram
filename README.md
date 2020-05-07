# BERTRAM (BERT for Attentive Mimicking)

This repository contains the code for [BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Representations](https://arxiv.org/abs/1910.07181). The paper introduces **BERTRAM**, a powerful architecture based on BERT that is capable of inferring high-quality embeddings for rare words that are suitable as input representations for deep language models. This is achieved by enabling the surface form and contexts of a word to interact with each other in a deep architecture.

## 📑 Contents

**[⚙️ Setup](#%EF%B8%8F-setup)**

**[💬 Usage](#-usage)**

**[💡 Training BERTRAM from Scratch](#-training-bertram-from-scratch)**

**[💾 Pre-Trained Models](#-pre-trained-models)**

**[📕 Citation](#-citation)**

## ⚙️ Setup

BERTRAM requires `Python>=3.7`, `jsonpickle`, `numpy`, `pytorch`, `torchvision`, `scipy`, `gensim`, `visdom` and `transformers==2.1`. If you use `conda`, you can simply create an environment with all required dependencies from the `environment.yml` file found in the root of this repository. 

## 💬 Usage

To use BERTRAM for downstream tasks, you can either [download a pretrained model](#-pre-trained-models) or [train your own instance of BERTRAM](#-training-bertram-from-scratch). Note that each instance of BERTRAM can only be used in combination with the pretrained transformer model for which it was trained.

To use a pretrained BERTRAM instance, first initialize a `BertramWrapper` object as follows:

```python
bertram = BertramWrapper('../models/bertram-add-for-bert-base-uncased', device='cpu')
```

You can infer embeddings for words from their surface-form and a (possibly empty) list of contexts using BERTRAM as follows:

```python
word = 'kumquat'
contexts =  ['litchi, pineapple and kumquat is planned for the greenhouse.', 'kumquat and cranberry sherbet']
bertram.infer_vector(word, contexts)
```

To directly inject a BERTRAM vector into a language model, you can use the `add_word_vectors_to_model()` method:

```python
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

words_with_contexts = {
    'kumquat': ['litchi, pineapple and kumquat is planned for the greenhouse.', 'kumquat and cranberry sherbet'],
    'resigntaion': []
}
bertram.add_word_vectors_to_model(words_with_contexts, tokenizer, model)
```

For each word `w` in the `words_with_contexts` dictionary, this adds a new token `<BERTRAM:w>` to the `tokenizer`'s vocabulary and adds the corresponding BERTRAM vector to the `model`'s embedding matrix. This way, the language model's original representation of `w` does not get lost. You can now represent each word `w` in various ways:

```python
input_standard = 'A kumquat is a [MASK]'                     # this uses the LM's default representation of 'kumquat'
input_bertram  = 'A <BERTRAM:kumquat> is a [MASK]'           # this uses only the BERTRAM vector for 'kumquat'
input_slash    = 'A kumquat / <BERTRAM:kumquat> is a [MASK]' # this uses both representations
```

In our experiments, we found the last variant (also called `BERTRAM-slash` in the paper) to perform best. A more detailed example can be found in `examples/use_bertram_for_mlm.py`.

## 💡 Training BERTRAM from Scratch

As described in the paper, training a new BERTRAM instance requires the following steps: (1) training a context-only model, (2) training a form-only model, (3) combining both models and training the combined model. 

### Preparing a Training Corpus

Before training a BERTRAM model, you need (1) a large plain-text file and (2) a set of target vectors that BERTRAM is trained to mimic. 

#### Handling the Plain-Text File

The plain-text file needs to be preprocessed using the script found [here](https://github.com/timoschick/form-context-model) as follows:
```
python3 fcm/preprocess.py train --input $PATH_TO_YOUR_TEXT_CORPUS --output $TRAIN_DIR
```
This creates various files in `$TRAIN_DIR`; the important ones are `train.vwc100` and all files of the form `train.bucket<X>`. The former contains words and their number of occurrences and is used by BERTRAM to build an *n*-gram vocabulary. The latter are used to generate contexts for training. Move all `train.bucket<X>` files into a separate folder `/buckets` inside `$TRAIN_DIR`.

#### Obtaining Target Vectors

Training BERTRAM requires a file `$EMBEDDING_FILE` where each line is of the form `<word> <embedding>`. You can initialize this file simply by iterating over the entire (uncontextualized) embedding matrix of a pretrained language model (an example for `bert-base-uncased` can be found [here](https://www.cis.uni-muenchen.de/~schickt/embeddings-bert-base-uncased.txt)). Note that the training procedure described in the paper makes use of [One-Token-Approximation](https://github.com/timoschick/one-token-approximation) to also obtain embeddings for frequent *multi-token* words; these embeddings are used as additional training targets.

### Training a Context-Only Model

Use the following command to train a context-only BERTRAM model:

```
python3 train_bertram.py \
   --model_cls $MODEL_CLS \
   --bert_model $MODEL_NAME \
   --output_dir $CONTEXT_OUTPUT_DIR \
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --num_train_epochs 5 \
   --emb_dim $EMB_DIM \
   --max_seq_length $MAX_SEQ_LENGTH \
   --mode context \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --no_finetuning \
   --smin 4 \
   --smax 32
```
where
- `$MODEL_CLS` is the class of the underlying language model (either `bert` or `roberta`)
- `$MODEL_NAME` is the name of the underlying language model (e.g., `bert-base-uncased`, `roberta-large`)
- `$CONTEXT_OUTPUT_DIR` is the output directory for the context-only model
- `$TRAIN_DIR` is the training dir from the previous step
- `$EMBEDDING_FILE` is the embedding file from the previous step
- `$EMB_DIM` is the word embedding dimension of the target vectors (e.g., `768` for `bert-base-uncased`)
- `$MAX_SEQ_LENGTH` is the maximum token length for each context
- `$TRAIN_BATCH_SIZE` is the batch size to be used during training

### Training a Form-Only Model

Use the following command to train a form-only BERTRAM model:

```
python3 train_bertram.py \
   --model_cls $MODEL_CLS \
   --bert_model $MODEL_NAME \
   --output_dir $FORM_OUTPUT_DIR \
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --num_train_epochs 20 \
   --emb_dim $EMB_DIM \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --smin 1 \
   --smax 1 \
   --max_seq_length 10 \
   --mode form \
   --learning_rate 0.01 \
   --dropout 0.1 \
```
where `$MODEL_CLS`, `$MODEL_NAME`, `$TRAIN_DIR`, `$EMBEDDING_FILE`, `$EMB_DIM` and `$TRAIN_BATCH_SIZE` are as for the context-only model and `$FORM_OUTPUT_DIR` is the output directory for the form-only model.

### Combining Both Models

Fuse both models as follows:

```
python3 fuse_models.py \
   --form_model $FORM_OUTPUT_DIR \
   --context_model $CONTEXT_OUTPUT_DIR \
   --mode $MODE \
   --output $FUSED_DIR
```
where `$FORM_OUTPUT_DIR` and `$CONTEXT_OUTPUT_DIR` are as before, `$MODE` is the configuration for the fused model (either `add` or `replace`) and `$FUSED_DIR` is the output directory for the fused model.

The fused model can then be trained as follows:

```
python3 train_bertram.py \
   --model_cls $MODEL_CLS \
   --bert_model $FUSED_DIR \
   --output_dir $OUTPUT_DIR \ 
   --train_dir $TRAIN_DIR/buckets/ \
   --vocab $TRAIN_DIR/train.vwc100 \
   --emb_file $EMBEDDING_FILE \
   --emb_dim $EMB_DIM \
   --mode $MODE \
   --train_batch_size $TRAIN_BATCH_SIZE \
   --max_seq_length $MAX_SEQ_LENGTH \
   --num_train_epochs 3 \
   --smin 4 \
   --smax 32 \
   --optimize_only_combinator 
```
where `$MODEL_CLS`, `$FUSED_DIR`, `$TRAIN_DIR`, `$EMBEDDING_FILE`, `$EMB_DIM`, `$MODE`, `$MAX_SEQ_LENGTH` and `$TRAIN_BATCH_SIZE` are as before and `$OUTPUT_DIR` is the output directory for the final model.

## 💾 Pre-Trained Models

🚨 All pre-trained BERTRAM models released here were trained on significantly less data than BERT/RoBERTa (6GB vs 16GB/160GB). To get better results for downstream task applications, consider [training your own instance of BERTRAM](#-training-bertram-from-scratch).

| BERTRAM Model Name                  | Configuration | Corresponding LM    | Link |
| :---------------------------------- | :------------ | :------------------ | :--- |
| `bertram-add-for-bert-base-uncased` | `ADD`         | `bert-base-uncased` | [📥 Download](https://www.cis.uni-muenchen.de/~schickt/bertram-add-for-bert-base-uncased.zip) |
| `bertram-add-for-roberta-large`     | `ADD`         | `roberta-large`     | [📥 Download](https://www.cis.uni-muenchen.de/~schickt/bertram-add-for-roberta-large.zip)

## 📕 Citation

If you make use of the code in this repository, please cite the following paper:

    @inproceedings{schick2020bertram,
      title={{BERTRAM}: Improved Word Embeddings Have Big Impact on Contextualized Representations},
      author={Schick, Timo and Sch{\"u}tze, Hinrich},
      url={https://arxiv.org/abs/1910.07181},
      booktitle={Proceedings of the 2020 Annual Conference of the Association for Computational Linguistics (ACL)},
      year={2019}
    } 