import pandas as pd
import numpy as np
import evaluate
import argparse
import pickle
from artelingo.evaluation.pycocoevalcap import Cider
import pdb
import jieba
from transformers import AutoTokenizer

checkpoint = "bigscience/bloomz-7b1-mt"
llama_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
llama_tokenizer.pad_token = llama_tokenizer.eos_token

def add_space(cap):
    # cap = jieba.cut(cap, use_paddle=True)
    cap = llama_tokenizer.tokenize(cap)
    cap = ' '.join(list(cap))
    return cap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=str, 
                        default='/ibex/user/mohameys/artelingo/test_metadata.csv',
                        help='Path to GT captions file')
    parser.add_argument('--generations', type=str, required=True,
                        help='Path to generated captions file')
    # parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--metrics-list', nargs='+', 
                        default=[
                                 'bleu',
                                 'google_bleu',
                                 'meteor',
                                # #  'sacrebleu', 
                                # #  'bleurt',
                                 'rouge',
                                 'cider',
                                 ])
    return parser.parse_args()

def group_references(references):
    references = references.drop(columns=['language', 'split'])
    group = references.groupby(['art_style', 'painting']).agg({
        'caption': list,
        'emotion': list
    })
    group.reset_index(inplace=True)
    return group

def dataframes_to_coco_eval_format(references, hypothesis):
    references = {i: [k for k in x] for i, x in enumerate(references)}
    hypothesis = {i: [x] for i, x in enumerate(hypothesis)}
    return references, hypothesis

def main():
    args = parse_args()
    metrics_list = args.metrics_list
    # language = args.language
    print('Loading references')
    references_all = pd.read_csv(args.references)
    print('Done\nLoading generations')
    generations_all = pd.read_csv(args.generations)
    generations_all = generations_all[generations_all['caption'].notna()]
    # for language in ['english', 'arabic', 'chinese']:
    print(f'================== "ALL" ==================')
    references = references_all
    grouped_references = group_references(references)
    generations = generations_all
    print('Done\nMerging')
    # pdb.set_trace()
    merged = pd.merge(generations, grouped_references, on=['art_style', 'painting'])

    hypothesis = merged['caption_x']
    references = merged['caption_y']
    hypothesis = hypothesis.apply(add_space)
    references = references.apply(lambda x: [add_space(sent) for sent in x])

    metrics = {m:evaluate.load(m) if m != 'cider' else None for m in metrics_list}
    print('Done\nComputing scores')
    final_scores = {}
    for m, metr in metrics.items():
        try:
            if m == 'cider':
                references2, hypothesis2 = dataframes_to_coco_eval_format(references, hypothesis)
                scorer = Cider()
                _, all_scores = scorer.compute_score(references2, hypothesis2)
                scores = pd.Series(all_scores).describe()[['mean', 'std']]
            elif m in ['rouge']:
                scores = metr.compute(predictions=hypothesis, references=references, tokenizer=lambda x: x.split())
            else:
                scores = metr.compute(predictions=hypothesis, references=references)
            print(f'******************* {m} *******************')
            print(scores)
            print('******************************************')
            
            final_scores[m] = scores
        except Exception as e:
            print(f'Error in {m}: {e}')

    # with open('scores.pkl', 'wb') as f:
    #     pickle.dump(final_scores, f)

if __name__ == '__main__':
    main()



