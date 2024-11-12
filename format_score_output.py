import re
from collections import defaultdict
import pandas as pd
import glob

def extract_scores(code, src_language):
    scores = defaultdict(list)
    matches = re.findall(r"'(.*?)': (.*?)[,|}]", code)
    for match in matches:
        key = match[0]
        value = match[1]
        if key in ['bleu', 'google_bleu', 'meteor', 'rougeL']:
            scores[key].append(float(value))
    
    matches = re.findall(r'mean\s+(.*?)\n', code)
    for match in matches:
        scores['cider'].append(float(match))
    matches = re.findall(r'=+\s+(.*?)\s+=+', code)
    for match in matches:
        scores['tgt_language'].append(match)
        scores['src_language'].append(src_language)
    
    return scores

# file_paths = glob.glob('/home/mohameys/minigpt4/results/fewshot/*.txt')
# file_paths = ['/home/mohameys/minigpt4/results/zeroshot/instructblip.txt'] 
# file_paths = ['/home/mohameys/minigpt4/results/zeroshot/laionx_artelingo.txt']
file_paths = ['/home/mohameys/minigpt4/results/zeroshot/clipcap_001.txt']
dfs = []
for fp in file_paths:
    src_language = fp.split('/')[-1].split('.')[0]
    code = open(fp, 'r').read()
    scores = extract_scores(code, src_language)
    dfs.append(pd.DataFrame(scores))
df = pd.concat(dfs)
# multiply by 100
df['bleu'] = df['bleu'] * 100
df['google_bleu'] = df['google_bleu'] * 100
df['meteor'] = df['meteor'] * 100
df['rougeL'] = df['rougeL'] * 100
df['cider'] = df['cider'] * 100
print(df.round(3))
df.round(3).to_csv('clipcap_001.csv', index=False)