import pandas as pd
import csv
import os
import json


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties

def load_evidence_type_dict(prop, model_name):
    
    evidence_type_dict = dict()
    dir_corpus = f'../analysis/{model_name}/annotation-tfidf-top_3_3-raw-10000-categories'

    with open(f'{dir_corpus}/{prop}/annotation-updated-done.csv') as infile:
        data = list(csv.DictReader(infile))
    for d in data:
        t = d['evidence_type']
        c = d['context']
        evidence_type_dict[c] = t
    return evidence_type_dict

def raw_to_distance(df, score_names, reference_name = 'median', percent =  False): 
    
    df_dict = df.to_dict('index')
    df_dict_distance = dict()
    
    median_dict = df_dict[reference_name]
    #score_names = ['prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    
    for i, d in df_dict.items():
        d_distance = dict()
        for k, v in d.items():
            if k in score_names:
                median = median_dict[k]
                dist =  v -  median
                # distnace in percent:
                dist_p = dist/median
                if percent == False:
                    d_distance[k] = dist
                else:
                    d_distance[k] = dist_p
            else:
                d_distance[k] = v
            df_dict_distance[i] = d_distance
    df_dict_distance[reference_name+'-reference'] = median_dict
    df_dist = pd.DataFrame(df_dict_distance).T
    return df_dist


def get_examples(model_name, prop, label):
    
    concepts_pos = set()
    # use 'all' category
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_all_pos = f'{dir_path}/{prop}/all/{label}/'
    
    for f in os.listdir(path_all_pos):
        concept = f.split('.')[0]
        if concept != '':
            concepts_pos.add(concept)
    return concepts_pos


def get_categories_concept(prop, concept, model_name):
    
    categories = set()
    
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_prop = f'{dir_path}/{prop}/'
    labels = ['pos', 'neg']
    
    for cat in os.listdir(path_prop):
        for label in labels:
            full_path = f'{path_prop}/{cat}/{label}/{concept}.csv'
            if os.path.isfile(full_path):
                categories.add(cat)
    return categories


def load_prop_data(prop):
    
    path = f'../data/aggregated_semantic_info_scalar/{prop}.json'
    with open(path) as infile:
        concept_dict = json.load(infile)
    return concept_dict