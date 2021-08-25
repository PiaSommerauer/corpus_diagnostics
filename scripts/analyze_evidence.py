from analyze_annotations import get_evidence_dict

import json
import csv
import os
from collections import Counter, defaultdict


def load_prop_data(prop):
    
    path = f'../data/aggregated_semantic_info/{prop}.json'
    with open(path) as infile:
        concept_dict = json.load(infile)
    return concept_dict


def load_concept_evidence(concept, prop, model_name, categories):
    
    categories.add('all')
    contexts = set()
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    
    for cat in categories:
        f_path = f'{dir_path}/{prop}/{cat}/pos/{concept}.csv'
        if os.path.isfile(f_path):
            with open(f_path) as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['']
                diff = float(d['diff'])
                if diff > 0:
                    contexts.add(context)
    return contexts  

def get_categories(prop, model_name):
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    categories = set()
    for d in os.listdir(path_dir):
        if '.' not in d:
            categories.add(d)
    return categories

def get_n_examples_cat(category, prop, model_name):
    
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}/{category}'
    
    n_example_dict = dict()
    for l in ['pos', 'neg']:
        path = f'{path_dir}/{l}'
        examples = os.listdir(path)
        examples = [f for f in examples if f.endswith('.csv')]
        n_example_dict[l] = len(examples)
    return n_example_dict

def get_top_ev_categories(prop, model_name, top_cutoff, concept_cutoff):
    #table = dict()
    aggregation_name = f'aggregated-tfidf-raw-10000-categories'
    categories = get_categories(prop, model_name)
    
    path_dir_agg = f'../analysis/{model_name}/{aggregation_name}/{prop}'
    evidence_dict = get_evidence_dict(model_name, prop, top_cutoff, concept_cutoff)
    
    et_context_dict = defaultdict(set)
    for c, et in evidence_dict.items():
        et_context_dict[et].add(c)
    et_sorted = ['p', 'n', 'l', 'i', 'r', 'b', 'u'] 
    et_cat_context_perf_dict = defaultdict(dict)
    
    # get top performance per evidence type for each category
    for cat in categories:
        path = f'{path_dir_agg}/{cat}.csv'
        n_example_dict = get_n_examples_cat(cat, prop, model_name)
        # load file containing all concepts
        with open(path) as infile:
            data = list(csv.DictReader(infile))
        n_pos = int(data[0]['total_pos'])
        n_neg = int(data[0]['total_neg'])
        if n_pos > 9 and n_neg > 9:
            perf_data = defaultdict(list)
            for d in data:
                f1 = float(d['f1'])
                perf_data[f1].append(d)
            perf_ranked =sorted(list(perf_data.keys()), reverse=True)
        
            # go through evidence types
            for et in et_sorted:
                contexts = et_context_dict[et]
                for f1 in perf_ranked:
                    data = perf_data[f1]
                    contexts_f1 = [d['context'] for d in data]
                    contexts_et = set()
                    for c in contexts_f1:
                        if c in contexts:
                            contexts_et.add(c)
                    if len(contexts_et) > 0:
                        et_cat_context_perf_dict[(cat, et)]['f1'] = round(f1, 2)
                        et_cat_context_perf_dict[(cat, et)]['contexts'] = ' '.join(contexts_et)
                        et_cat_context_perf_dict[(cat, et)].update(n_example_dict)
                        break              
                
    return et_cat_context_perf_dict