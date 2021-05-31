import json

from tqdm import tqdm
import os
import sys
from collections import defaultdict

from numba import jit, cuda

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf


def load_data(prop):
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict

# get vocab to check what is in vocab

def load_vocab(path):
    vocab_counts = dict()
    with open (path) as infile:
        for line in infile:
            word, cnt = line.strip().split(' ')
            vocab_counts[word] = cnt
    return vocab_counts

def load_pairs(path):
    pairs = []
    with open(path) as infile:
        for line in infile:
            pair = line.strip().split(' ')
            pairs.append(pair)
    return pairs


def to_file(contexts, prop, target, label):
    
    path_dir = f'../contexts'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = f'../contexts/{prop}'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = f'../contexts/{prop}/{label}'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    contexts_str = ' '.join(contexts)
    with open(f'{path_dir}/{target}.txt', 'w') as outfile:
        outfile.write(contexts_str)
        
@jit(target ="cuda")        
def extract_contexts(word_list, pair_path, prop, label, n_lines):
    
    path_dir = f'../contexts'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = f'../contexts/{prop}'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = f'../contexts/{prop}/{label}'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    with open(pair_path) as infile:
        contexts = infile.read().strip().split('\n')
    print('loaded contexts')
    
    target_contexts = defaultdict(list)
    for line in tqdm(contexts, total = len(contexts)):
        pair = line.strip().split(' ')
        w, c = pair
        for target in word_list:
            if target == w:
                target_contexts[target].append(c)
    for target, contexts in target_contexts.items():
        path_contexts = f'{path_dir}/{target}.txt'
        with open(path_contexts, 'w') as outfile:
            outfile.write(' '.join(contexts))
                        
                        
                        
def main():
    
    prop = sys.argv[1]
    path_dir = sys.argv[2]
    prop_dict = load_data(prop)

    target_pos = [k for k, d in prop_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some']]
    target_neg = [k for k, d in prop_dict.items() if d['ml_label'] in ['few']]
    
    #path_dir = '/Users/piasommerauer/Data/dsm/corpus_exploration/wiki_full/run.2017-03-20-10:31:44'
    #test_vocab = f'{test_path_dir}/counts.words.vocab'
    pair_path =  f'{path_dir}/pairs-orig'
    # for progress bar  - use wc -l on terminal to get line number of pairs file
    n_lines = 1530697448
    
    label = 'pos'
    extract_contexts(target_pos, pair_path, prop, label, n_lines)
    label = 'neg'
    extract_contexts(target_neg, pair_path, prop, label, n_lines)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
