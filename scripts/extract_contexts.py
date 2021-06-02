import json

from tqdm import tqdm
import os
import sys
from collections import defaultdict
import multiprocessing
from collections import defaultdict

import itertools

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



def extract_contexts(qu, prop_targets, pair_line):

    pair = pair_line.strip().split(' ')
    w, c = pair
    target_contexts= defaultdict(dict)
    for prop, label_dict in prop_targets.items():
        for label, targets in label_dict.items():
            target_contexts[prop][label] = defaultdict(list)
            for target in targets:
                if target == w:
                    target_contexts[prop][label][target].append(c)
    qu.put(target_contexts)



def get_batches(n_lines, n_batches):
    batch_size = round(n_lines / n_batches)
    batches = []
    start = 0
    for n in range(n_batches):
        end = start + batch_size
        batches.append((start, end))
        start = end
    # add final batch
    last_b = batches[-1]
    end = last_b[1]
    if end < n_lines:
        batches.append((end, n_lines + 1))
    return batches
                        
def main():


    prop = sys.argv[1]
    path_dir = sys.argv[2]
    model_name = sys.argv[3]
    pair_path =  f'{path_dir}/pairs-orig'

    # for progress bar  - use wc -l on terminal to get line number of pairs file
    #n_lines = 1530697448
    n_lines = 6
    #n_batches = 1000000
    n_batches = 1
    batches = get_batches(n_lines, n_batches)

    props = [prop]
    prop_targets = dict()
    all_contexts = dict()
    for prop in props:
        all_contexts[prop] = dict()
        prop_dict = load_data(prop)
        target_pos = [k for k, d in prop_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some']]
        target_neg = [k for k, d in prop_dict.items() if d['ml_label'] in ['few']]
        prop_targets[prop] = dict()
        prop_targets[prop]['pos'] = target_pos
        prop_targets[prop]['neg'] = target_neg
        all_contexts[prop]['pos'] = defaultdict(list)
        all_contexts[prop]['neg'] = defaultdict(list)

    batch_cnt = 0
    for start, end in batches:
        #output_dicts_batch = []
        with open(pair_path) as infile:
            my_lines =  itertools.islice(infile, start, end)
            jobs = []
            qu = multiprocessing.Queue()
            for pair_line in my_lines:
                p = multiprocessing.Process(target=extract_contexts,
                                            args=(qu, prop_targets, pair_line))
                jobs.append(p)
                p.start()

            for proc in jobs:
                out = qu.get()
                for prop, label_dict in out.items():
                    for label, target_dict in label_dict.items():
                        for target, contexts in target_dict.items():
                            all_contexts[prop][label][target].extend(contexts)
                proc.join()
        batch_cnt += 1
        print(all_contexts)
        if batch_cnt == 3:
            break



    for prop, label_dict in all_contexts.items():
        for label, target_dict in label_dict.items():
            path_dir = f'../contexts/{model_name}/{prop}/{label}'
            os.makedirs(path_dir, exist_ok=True)
            for target, contexts in target_dict.items():
                with open(f'{path_dir}/{target}.txt', 'w') as outfile:
                    outfile.write(' '.join(contexts))

if __name__ == '__main__':
    main()
    
    
    
    
