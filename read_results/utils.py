from Levenshtein import distance as lev_distance
import random
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import random
import numpy as np
import argparse
from paragraph2actions.readable_converter import ReadableConverter
import re
from transformers import AutoTokenizer
from collections import defaultdict
import time
from functools import wraps
import os
import torch
import textdistance
from typing import List

def levenshtein_similarity(truth: List[str], pred: List[str]) -> List[float]:
    assert len(truth) == len(pred)
    scores: List[float] = [
        textdistance.levenshtein.normalized_similarity(t, p)
        for t, p in zip(truth, pred)
    ]
    return scores

def modified_bleu(truth: List[str], pred: List[str], bleu_n=4) -> float:
    """
    Calculates the BLEU score of a translation, with a small modification in order not to penalize sentences
    with less than 4 words.

    Returns:
        value between 0 and 1.
    """
    references = [sentence.split() for sentence in truth]
    candidates = [sentence.split() for sentence in pred]

    # BLEU penalizes sentences with only one word. Even correct translations get a score of zero.
    references = [r + max(0, bleu_n - len(r)) * [""] for r in references]
    candidates = [c + max(0, bleu_n - len(c)) * [""] for c in candidates]

    # references must have a larger depth because it supports multiple choices
    refs = [[r] for r in references]
    weights = {
        2: (0.5, 0.5),
        4: (0.25, 0.25, 0.25, 0.25),
    }
    return 100*corpus_bleu(refs, candidates, weights=weights[bleu_n])  # type: ignore[no-any-return]

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} finished in {end_time - start_time:.5f} seconds.\n")
        return result
    return wrapper

def accuracy_score(score_list, threshold):
    matches = sum(score>=threshold for score in score_list)
    acc = matches / len(score_list)
    return acc

def extract_tokenized_entities(text):
    pattern = r'\$[^\$]+\$|#[^#]+#|@[^\@]+@'
    return re.findall(pattern, text)

def extract_reactant_cnt(text):
    max_id = None
    for token in text.split():
        if token.startswith('$') and token.endswith('$'):
            try:
                current_id = int(token.strip('$'))
                if max_id is None or current_id > max_id:
                    max_id = current_id
            except ValueError:
                pass  # Ignore tokens that do not represent an integer
    if not max_id:
        return 0
    return max_id

class Metric_calculator:
    def __init__(self, text_trunc_length=1024):
        self.converter = ReadableConverter(separator=' ; ')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=False, padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.text_trunc_length = text_trunc_length
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def tokenize(self, gt_list, pred_list):
        references = []
        hypotheses = []
        
        for gt, out in tqdm(zip(gt_list, pred_list)):
            gt_tokens = self.tokenizer.tokenize(gt)
            ## added for galactica
            gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = self.tokenizer.tokenize(out)
            out_tokens = list(filter(('<pad>').__ne__, out_tokens))
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            references.append([gt_tokens])
            hypotheses.append(out_tokens)
        return references, hypotheses
    
    @time_it
    def __call__(self, gt_list, pred_list, use_tokenizer=False):
        gt_list = [gt.strip() for gt in gt_list]
        pred_list = [pred.strip() for pred in pred_list]

        if use_tokenizer:
            references, hypotheses = self.tokenize(gt_list, pred_list)
            bleu2, bleu4 = self.bleu(references, hypotheses)
            _meteor_score = self.meteor(references, hypotheses)
        else:
            bleu2 = modified_bleu(gt_list, pred_list, bleu_n=2)
            bleu4 = modified_bleu(gt_list, pred_list, bleu_n=4)
            _meteor_score = 0
        rouge_1, rouge_2, rouge_l = self.rouge(gt_list, pred_list)

        validity = self.validity(gt_list, pred_list)
        acc_100, acc_90, acc_75, acc_50 = self.accuracy(gt_list, pred_list)

        print('BLEU-2 score:', bleu2)
        print('BLEU-4 score:', bleu4)
        print('Average Meteor score:', _meteor_score)
        print('rouge1:', rouge_1)
        print('rouge2:', rouge_2)
        print('rougeL:', rouge_l)
        
        print(f'Validity: {validity:.6f}')
        print(f'Accuracy (100): {acc_100:.6f}')
        print(f'Accuracy (90): {acc_90:.6f}')
        print(f'Accuracy (75): {acc_75:.6f}')
        print(f'Accuracy (50): {acc_50:.6f}')

        line = ''
        for score in [validity, bleu2, bleu4, acc_100, acc_90, acc_75, acc_50, rouge_1, rouge_2, rouge_l, _meteor_score]:
            line += f'{score:.6f} '
        print(line)
        
        return {
            'bleu2': bleu2,
            'bleu4': bleu4,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'meteor_score': _meteor_score,
            'validity': validity,
            'acc_100': acc_100,
            'acc_90': acc_90,
            'acc_75': acc_75,
            'acc_50': acc_50,
        }
    
    def get_result_list(self, gt_list, pred_list, use_tokenizer=False):
        gt_list = [gt.strip() for gt in gt_list]
        pred_list = [pred.strip() for pred in pred_list]

        if use_tokenizer:
            references, hypotheses = self.tokenize(gt_list, pred_list)
            bleu2 = [corpus_bleu([gt], [pred], weights=(.5,.5)) for gt, pred in zip(references, hypotheses)]
            bleu4 = [corpus_bleu([gt], [pred], weights=(.25,.25,.25,.25)) for gt, pred in zip(references, hypotheses)]
            _meteor_score = [meteor_score(gt, out) for gt, out in zip(references, hypotheses)]
        else:
            bleu2 = [modified_bleu([gt], [pred], bleu_n=2) for gt, pred in zip(gt_list, pred_list)]
            bleu4 = [modified_bleu([gt], [pred], bleu_n=4) for gt, pred in zip(gt_list, pred_list)]
            _meteor_score = 0
        rouge_1, rouge_2, rouge_l = self.rouge(gt_list, pred_list, return_list=True)

        lev_score = levenshtein_similarity(gt_list, pred_list)
        
        return {
            'bleu2': bleu2,
            'bleu4': bleu4,
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'meteor_score': _meteor_score,
            'lev_score': lev_score,
        }
    
    def bleu(self, references, hypotheses):
        bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
        bleu2 *= 100
        bleu4 *= 100
        return bleu2, bleu4
    
    def meteor(self, references, hypotheses):
        meteor_scores = []
        for gt, out in zip(references, hypotheses):
            mscore = meteor_score(gt, out)
            meteor_scores.append(mscore)
        _meteor_score = np.mean(meteor_scores)
        _meteor_score *= 100
        return _meteor_score

    def rouge(self, targets, predictions, return_list=False):
        rouge_scores = []
        for gt, out in zip(targets, predictions):
            rs = self.scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = [rs['rouge1'].fmeasure for rs in rouge_scores]
        rouge_2 = [rs['rouge2'].fmeasure for rs in rouge_scores]
        rouge_l = [rs['rougeL'].fmeasure for rs in rouge_scores]
        if return_list:
            return rouge_1, rouge_2, rouge_l

        rouge_1 = np.mean(rouge_1) * 100
        rouge_2 = np.mean(rouge_2) * 100
        rouge_l = np.mean(rouge_l) * 100
        return rouge_1, rouge_2, rouge_l
        

    def validity(self, gt_list, pred_list):
        num_valid, n = 0, len(pred_list)
        for pred, gt in zip(pred_list, gt_list):
            try:
                actions = self.converter.string_to_actions(pred)
                max_token_pred = extract_reactant_cnt(pred)
                max_token_gt = extract_reactant_cnt(gt)
                assert max_token_gt >= max_token_pred
                num_valid += 1
            except:
                pass
        return 100*(num_valid / n)
    
    def accuracy(self, gt_list, pred_list):
        score_list = levenshtein_similarity(gt_list, pred_list)
        acc_100 = 100*accuracy_score(score_list, 1.0)
        acc_90 = 100*accuracy_score(score_list, 0.90)
        acc_75 = 100*accuracy_score(score_list, 0.75)
        acc_50 = 100*accuracy_score(score_list, 0.50)
        return acc_100, acc_90, acc_75, acc_50
