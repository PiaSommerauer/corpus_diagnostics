from process_tfidf import get_properties
from collections import defaultdict
from nltk.corpus import wordnet as wn
import os
import csv
import json

def get_semantic_info_files():
    path = '../data/semantic_info/full/'
    collections = {'activities', 'perceptual', 'complex', 'parts'}
    prop_files = []
    for coll in collections:
        coll_path = f'{path}{coll}'
        for f in os.listdir(coll_path):
            full_path = f'{coll_path}/{f}'
            prop_files.append(full_path)
    return prop_files

def load_semantic_info(f):
    with open(f) as infile:
        line = infile.read().split('\n')[1]
        if '\t' in line:
            deli = '\t'
        else:
            deli = ','
    with open(f) as infile:
        data = list(csv.DictReader(infile, delimiter=deli))
    return data

def collect_category_data(prop,  semantic_data):
    cat_concepts_dict = defaultdict(set)
    for d in semantic_data:
        concept = d['lemma']
        categories = set(d['categories_str'].split(' '))
        prop_cats = [f'is_{prop}', prop]
        if len(categories) > 1:
            categories = [cat for cat in prop_cats if cat not in prop_cats]
        categories = '-'.join(categories)
        cat_concepts_dict[categories].add(concept)
    return cat_concepts_dict

def load_concepts(prop):
    prop_path = f'../data/aggregated/{prop}.json'
    with open(prop_path) as infile:
        concept_dict = json.load(infile)
    concepts_pos = [c for c, d in concept_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some', 'some']]
    concepts_neg = [c for c, d in concept_dict.items() if d['ml_label'] in ['few']]
    all_concepts = concepts_pos + concepts_neg
    return set(all_concepts), set(concepts_pos), set(concepts_neg)
    

def load_semantic_categories_original(properties):
    prop_cat_concepts_dict = defaultdict(dict)
    # get paths for semantic category info
    prop_files = get_semantic_info_files()
    
    for f in prop_files:
        prop = os.path.basename(f).split('.')[0]

        if prop in properties:
            all_concepts, concepts_pos, concepts_neg = load_concepts(prop)
            # load semantic info
            semantic_data = load_semantic_info(f)
            # collect category data:
            cat_concepts_dict = collect_category_data(prop, semantic_data)
                
            # Clean dict
            clean_cat_concepts_dict = defaultdict(set)
            concepts_sorted = set()
            for cat, concepts in cat_concepts_dict.items():
                concepts_in_data = all_concepts.intersection(concepts)
                concepts_sorted.update(concepts_in_data)
                #print(cat, 'concepts in data:', concepts_in_data)
                if cat.startswith('is_'):
                    cat = cat[3:]
                elif cat.startswith('does_'):
                    cat = cat[5:]
                elif cat == '' or cat == 'neighbors-200' or cat == 'pos' or cat == 'neg':
                    cat = prop
                clean_cat_concepts_dict[cat].update(concepts_in_data)
            # add concepts not sorted
            concepts_data_remaining = all_concepts.difference(concepts_sorted)
            clean_cat_concepts_dict['not_sorted'] = concepts_data_remaining

            # merge wings (wing) and wheels (wheel):
            if prop == 'wings':
                clean_cat_concepts_dict['wings'].update(clean_cat_concepts_dict['wing']) 
                clean_cat_concepts_dict['wing'] = set()
            if prop == 'wheels':
                clean_cat_concepts_dict['wheels'].update(clean_cat_concepts_dict['wheel']) 
                clean_cat_concepts_dict['wheel'] = set()

            # add total
            clean_cat_concepts_dict['all-pos'] = concepts_pos
            clean_cat_concepts_dict['all-neg'] = concepts_neg
            # remove empty cats:
            #print('clean dict:')
            clean_cat_concepts_dict_filtered = dict()
            for cat, concepts in clean_cat_concepts_dict.items():
                if len(concepts) != 0:
                    clean_cat_concepts_dict_filtered[cat] = concepts
            prop_cat_concepts_dict[prop] = clean_cat_concepts_dict_filtered
    return prop_cat_concepts_dict


def collect_search_log():
    path_log = '../data/semantic_info/candidate_extraction/log.json'
    with open(path_log) as infile:
        data = json.load(infile)
    cat_synset_dict = defaultdict(set)
    # sort synsets to categories
    for search_entry in data:
        source = search_entry['source']
        n_extracted = search_entry['n_extracted']
        cat = search_entry['search_term']
        extracted_cats = search_entry['extracted_categories']
        if source == 'wn_hypernym' and n_extracted != 's.o.' and len(extracted_cats) > 0:
                cat_synset_dict[cat].update(extracted_cats)
    return cat_synset_dict

def get_hyponym_lemmas(syn):
   
    all_lemmas = set()
    all_hyponyms = [syn]
    for hyp in all_hyponyms:
        new_hyps = hyp.hyponyms()
        all_hyponyms.extend(new_hyps)
    for syn in all_hyponyms:
        lemmas = syn.lemmas()
        lemmas_str = [lemma.name() for lemma in lemmas]
        all_lemmas.update(lemmas_str)
    return all_lemmas

def sort_wn(cat, synsets, concepts_original):
    concept_syn_dict = defaultdict(set)
    for syn_str in synsets:
        if syn_str.startswith('Synset'):
            syn_id = syn_str[8:].split("'")[0]
            syn = wn.synset(syn_id) 
        else:
            synsets = wn.synsets(cat)
            index = int(syn_str)
            syn = synsets[index]
            syn_str = str(syn)
        lemmas = get_hyponym_lemmas(syn)
        concepts_in_cat = set(lemmas).intersection(concepts_original)
        for concept in concepts_in_cat:
            concept_syn_dict[concept].add(syn_str)
    return concept_syn_dict


def update_semantic_categories(prop, semantic_categories_original, cat_synsets):
    concept_category_synsets = defaultdict(dict)
    concept_original_cat =  defaultdict(list)
    cats_all = ['all-pos', 'all-neg']
    cat_concepts_original = semantic_categories_original[prop]

    # update original categories
    for cat_original, concepts in cat_concepts_original.items():
        if cat_original not in cats_all:
            # sort into updated categories
            for cat, synsets in cat_synsets.items():
                #print(cat_original, cat, synsets)
                concept_syn_dict = sort_wn(cat, synsets, concepts)
                for concept, syns in concept_syn_dict.items():
                    concept_category_synsets[concept][cat] = list(syns)

    # check for remaining unsorted concepts:
    all_pos = cat_concepts_original['all-pos']
    all_neg = cat_concepts_original['all-neg']
    all_concepts = all_pos.union(all_neg)
    all_sorted = set(concept_category_synsets.keys())
    unsorted = all_concepts.difference(all_sorted)
    for concept in unsorted:
        concept_category_synsets[concept][prop] = ['unsorted']
    return concept_category_synsets


def add_categories_to_data(prop, concept_category_synsets):
    
    # new dir
    dir_path = '../data/aggregated_semantic_info/'
    os.makedirs(dir_path, exist_ok=True)
    
    # path_original:
    path_original = f'../data/aggregated/{prop}.json'
    path_updated =  f'../data/aggregated_semantic_info/{prop}.json'
    # load original_file:
    with open(path_original) as infile:
        concept_dict = json.load(infile)
    for concept, d in concept_dict.items():
        cat_dict = concept_category_synsets[concept]
        d['categories']  = cat_dict
    # updated dict to new file
    with open(path_updated, 'w') as outfile:
        json.dump(concept_dict,  outfile)
        
        
def main():
    properties = get_properties()
    # get already existing categories
    semantic_categories_original = load_semantic_categories_original(properties)
    # get original search log info for wordnet synsets used for categories
    cat_synsets = collect_search_log()
    
    for prop in properties:
        concept_category_synsets = update_semantic_categories(prop, semantic_categories_original, cat_synsets)
        add_categories_to_data(prop, concept_category_synsets)
        

if __name__ == '__main__':
    main()

    
    