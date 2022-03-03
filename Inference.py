import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import collections
import numpy as np
import torch
from argparse import ArgumentParser
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_qa_inputs(question, context, tokenizer):
    return tokenizer.encode_plus(question, context, return_tensors='pt')

def get_clean_text(tokens, tokenizer):
    text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
        )
    text = text.strip()
    text = " ".join(text.split())
    return text

def prediction_probabilities(predictions):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    all_scores = [pred.start_logit+pred.end_logit for pred in predictions] 
    return softmax(np.array(all_scores))

def preliminary_predictions(start_logits_, end_logits_, input_ids, nbest):
    start_logits = to_list(start_logits_)[0]
    end_logits = to_list(end_logits_)[0]
    tokens = to_list(input_ids)[0]

    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
    
    start_indexes = [idx for idx, logit in start_idx_and_logit[:nbest]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:nbest]]

    question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(102)])]

    PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index = start_index,
                    end_index = end_index,
                    start_logit = start_logits[start_index],
                    end_logit = end_logits[end_index]
                )
            )
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    return prelim_preds, (tokens, start_logits, end_logits)


def best_predictions(prelim_preds, nbest, tok_logits, tokenizer):
    tokens, start_logits, end_logits = tok_logits
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )
    nbest_predictions = []
    seen_predictions = []
    for pred in prelim_preds:
        if len(nbest_predictions) >= nbest: 
            break
        if pred.start_index > 0:
            toks = tokens[pred.start_index : pred.end_index+1]
            text = get_clean_text(toks, tokenizer)

            if text in seen_predictions:
                continue

            seen_predictions.append(text) 

            nbest_predictions.append(
                BestPrediction(
                    text=text, 
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit
                    )
                )
        
    nbest_predictions.append(
        BestPrediction(
            text="", 
            start_logit=start_logits[0], 
            end_logit=end_logits[0]
            )
        )
    return nbest_predictions


def compute_score_difference(predictions):
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null


def get_robust_prediction(question, context, model, tokenizer, nbest=10, null_threshold=1.0):
    
    inputs = get_qa_inputs(question, context, tokenizer)
    start_logits, end_logits = model(**inputs, return_dict=False)

    prelim_preds, tok_logits = preliminary_predictions(start_logits, 
                                           end_logits, 
                                           inputs['input_ids'],
                                           nbest)
    
    nbest_preds = best_predictions(prelim_preds, nbest, tok_logits, tokenizer)

    probabilities = prediction_probabilities(nbest_preds)
        
    score_difference = compute_score_difference(nbest_preds)

    if score_difference > null_threshold:
        return "#NO_ANSWER#", probabilities[-1]
    else:
        return nbest_preds[0].text, probabilities[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    text = args.text
    question = args.question

    output_dir = 'model'
    do_lowercase = True

    config = AutoConfig.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, do_lower_case=do_lowercase, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(output_dir, config=config)

    answer, prob = get_robust_prediction(question, text, model, tokenizer, nbest=10, null_threshold=0)
    print('Question: {}'.format(question))
    print('Answer: {}\n'.format(answer))

if __name__ == '__main__':
    main()