from typing import List

import torch
from transformers import BertForMaskedLM, BertTokenizer, GPT2Tokenizer

from bertram import BertramWrapper


def predict(inp: str, model: BertForMaskedLM, tokenizer: BertTokenizer, k: int = 3) -> List[str]:
    """
    Predict the top-k substitutes for an input text containing a single MASK token.
    :param inp: the input text
    :param model: a masked language model
    :param tokenizer: the tokenizer corresponding to the model
    :param k: the number of predictions
    :return: the list of top-k substitutes for the MASK token
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    input_ids = tokenizer.encode(inp, add_special_tokens=True, **kwargs)
    mask_idx = input_ids.index(tokenizer.mask_token_id)
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        (predictions,) = model(input_ids)

    predicted_tokens = []
    _, predicted_indices = torch.topk(predictions[0, mask_idx], k)

    for predicted_index in predicted_indices:
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index.item()])[0]
        predicted_tokens.append(predicted_token)
    return predicted_tokens


def main():
    # load a pre-trained BERTRAM model and the corresponding BERT model
    bertram = BertramWrapper('../models/bertram-add-for-bert-base-uncased', device='cpu')
    bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    words_with_contexts = {
        'kumquat': ['litchi, pineapple and kumquat is planned for the greenhouse.', 'kumquat and cranberry sherbet'],
        'resigntaion': []
    }

    # infer a BERTRAM vector for a single word from it's surface form and contexts
    print(f'BERTRAM vector for "kumquat": {bertram.infer_vector("kumquat", words_with_contexts["kumquat"])[:5]}')

    # infer BERTRAM vectors for all words and add them to the transformer's embedding matrix
    # for each word `w`, this creates a new token `<BERTRAM:w>` that can be used like a regular word
    bertram.add_word_vectors_to_model(words_with_contexts, tokenizer, bert)

    inputs_bert = ["a kumquat is a [MASK].", "'resigntaion' is a misspelling of '[MASK]'."]
    inputs_bertram = ["a <BERTRAM:kumquat> is a [MASK].", "'<BERTRAM:resigntaion>' is a misspelling of '[MASK]'."]

    for input_bert, input_bertram in zip(inputs_bert, inputs_bertram):
        bert_predictions = predict(input_bert, bert, tokenizer)
        bertram_predictions = predict(input_bertram, bert, tokenizer)
        print(f'Input: {input_bert} \n\tBERT:    {bert_predictions}\n\tBERTRAM: {bertram_predictions}\n')


if __name__ == '__main__':
    main()
