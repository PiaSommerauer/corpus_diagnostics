import json

from tqdm import tqdm
import os
import sys
from collections import defaultdict
#from numba import jit

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf


def load_data(prop):
    path = '../data/aggregated/'+prop+'.json'
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
    
    path_dir = '../contexts'
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = '../contexts/'+prop
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    path_dir = '../contexts/'+prop+'/'+label
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)
    contexts_str = ' '.join(contexts)
    with open(path_dir+'/'+target+'.txt', 'w') as outfile:
        outfile.write(contexts_str)


def extract_contexts(prop_targets, pair_line, target_contexts):

    pair = pair_line.strip().split(' ')
    w, c = pair
    for prop, label_dict in prop_targets.items():
        for label, targets in label_dict.items():
            target_contexts[prop][label] = dict()
            for target in targets:
                if target == w:
                    if target in target_contexts[prop][label]:
                        target_contexts[prop][label][target].append(c)
                    else:
                        target_contexts[prop][label][target] = [c]

                        
                        
                        
def main():
    
    #prop = sys.argv[1]
    prop = sys.argv[1]
    path_dir = sys.argv[2]


    #path_dir = '/Users/piasommerauer/Data/dsm/corpus_exploration/wiki_full/run.2017-03-20-10:31:44'
    #test_vocab = f'{test_path_dir}/counts.words.vocab'
    pair_path =  f'{path_dir}/pairs-orig'

    # for progress bar  - use wc -l on terminal to get line number of pairs file
    n_lines = 1530697448



    props = [prop]
    labels = ['pos', 'neg']
    prop_targets = dict()
    target_contexts = dict()
    for prop in props:
        prop_dict = load_data(prop)
        target_pos = [k for k, d in prop_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some']]
        target_neg = [k for k, d in prop_dict.items() if d['ml_label'] in ['few']]
        prop_targets[prop] = dict()
        target_contexts[prop] = dict()
        prop_targets[prop]['pos'] = target_pos
        prop_targets[prop]['neg'] = target_neg
        for label in labels:
            target_contexts[prop][label] = dict()

    with open(pair_path) as infile:
        for pair_line in tqdm(infile, total=n_lines):
            extract_contexts(prop_targets, pair_line, target_contexts)

    path_dir = '../contexts/' + prop + '/' + label
    os.makdirs(path_dir, exist_ok=True)
    for prop, label_dict in target_contexts.items():
        for label, target_dict in label_dict.items():
            for target, contexts in target_dict.items():
                path_dir = '../contexts/' + prop + '/' + label
                os.makdirs(path_dir, exist_ok=True)
                with open(path_dir+'/'+target+'.txt', 'w') as outfile:
                    outfile.write(' '.join(contexts))

if __name__ == '__main__':
    main()
    
    
    
    
