import os
import csv
from collections import Counter, defaultdict
import json
import pandas as pd



def get_n_concepts_total(path_config):
    with open(path_config) as infile:
        lines = infile.read().strip().split('\n')
    line = lines[1]
    n = int(line.split(': ')[1])
    return n

def load_annotations(prop, model, category, max_features):
    
    dir_annotation = f'../analysis/{model}/annotation-tfidf-top20-raw-{max_features}'
    dir_scores = f'../analysis/{model}/tfidf_aggregated_concept_scores-raw-{max_features}'
    path_pos = f'{dir_annotation}/{prop}-pos/{category}/{prop}-pos-annotated.csv'
    path_neg_full = f'{dir_scores}/{prop}-neg.csv'
    path_pos_full = f'{dir_scores}/{prop}-pos/{category}/{prop}-pos.csv'

    with open(path_pos) as infile:
        data_pos = list(csv.DictReader(infile))
    with open(path_neg_full) as infile:
        data_neg_full = list(csv.DictReader(infile))
    with open(path_pos_full) as infile:
        data_pos_full = list(csv.DictReader(infile))
           
    neg_dict = dict()
    for n, d in enumerate(data_neg_full, 1):
        word = d['context']
        d['rank'] = n
        if n > 19:
            d['top_20'] = False
        else:
            d['top_20'] = True
        if word not in neg_dict:
            neg_dict[word] = d
    pos_dict = dict()
    for n, d in enumerate(data_pos_full, 1):
        word = d['context']
        d['rank'] = n
        if n > 19:
            d['top_20'] = False
        else:
            d['top_20'] = True
        if word not in pos_dict:
            pos_dict[word] = d
            
    ev_dict_pos = dict()
    ev_dict_neg = dict()
    for d in data_pos:
        word = d['context']
        rank = pos_dict[word]['rank']
        d['rank'] = rank
        if word not in ev_dict_pos:
            ev = d['evidence']
            if n > 19:
                in_top_10=False
            else:
                in_top_10=True
            d['top_20'] = in_top_10
            if ev  in {'p', 'n', 'r', 'i', 'b'}:
                d['evidence_type']  = ev
                ev_dict_pos[word] = d
                if word in neg_dict:
                    d_neg = neg_dict[word]
                    ev_dict_neg[word]=d_neg
    # make combinations
    return ev_dict_pos, ev_dict_neg
 
def get_evidence_table(prop, model, category, max_features):
    ev_dict_pos, ev_dict_neg = load_annotations(prop, model, category, max_features)
    path_dir = f'../analysis/{model}/tfidf_aggregated_concept_scores-raw-{max_features}'
    path_neg_config = f'{path_dir}/{prop}-neg-config.txt'
    path_pos_config = f'{path_dir}/{prop}-pos/{category}/{prop}-pos-config.txt'
    n_pos = get_n_concepts_total(path_pos_config)
    n_neg = get_n_concepts_total(path_neg_config)
    print('n pos:', n_pos)
    print('n neg:', n_neg)
    print()
    table_dicts = []
    for w, d_pos in ev_dict_pos.items():
        d_table = dict()
        d_table['evidence'] = w
        d_table['evidence_type']  = d_pos['evidence_type']
        d_table['n_pos'] = round(int(d_pos['n_concepts'])/n_pos, 2)
        d_table['mean_tfidf_pos'] = float(d_pos['mean_tfidf'])
        d_table['rank_pos'] = d_pos['rank']
        #d_table['top_10_pos'] = d_pos['top_10']
        if w in ev_dict_neg:
            d_table['n_neg'] = round(int(ev_dict_neg[w]['n_concepts'])/n_neg, 2)
            d_table['mean_tfidf_neg'] = float(ev_dict_neg[w]['mean_tfidf'])
            d_table['rank_neg'] = ev_dict_neg[w]['rank']
            #d_table['top_10_neg'] = ev_dict_neg[w]['top_10']
        else:
            d_table['n_neg'] = '-'
            d_table['mean_tfidf_neg'] = ''
            d_table['rank_neg'] = ''
            #d_table['top_10_neg'] = ''
        table_dicts.append(d_table)
    return table_dicts 


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties



def get_concept_context_matrix(prop, model):
    path_pos = f'../analysis/{model}/annotation-tfidf-top20-raw/{prop}-pos-annotated.csv'
    path_all_pos_dir = f'../results/{model}/tfidf-raw/each_target_vs_corpus/{prop}/pos'
    path_all_neg_dir = f'../results/{model}/tfidf-raw/each_target_vs_corpus/{prop}/neg'
    
    # get evidence words:
    with open(path_pos) as infile:
        data = list(csv.DictReader(infile))
    evidence_words  = [d['context'] for d in data if d['evidence']=='yes']
    
    paths_pos = [f'{path_all_pos_dir}/{f}' for f in os.listdir(path_all_pos_dir) if f.endswith('.csv')]
    paths_neg = [f'{path_all_neg_dir}/{f}' for f in os.listdir(path_all_neg_dir) if f.endswith('.csv')]
    
    # get tf_idf_values
    concept_ev_dict = dict()
    concept_label_dict = dict()
    for f in paths_pos+paths_neg:
        concept = os.path.basename(f).split('.')[0]
        if f in paths_pos:
            concept_label_dict[concept] = 'pos'
        elif f in paths_neg:
            concept_label_dict[concept] = 'neg'
        with open(f) as infile:
            data = list(csv.DictReader(infile))
        context_score_dict = dict()
        for d in data:
            word = d['']
            score = float(d['target'])
            diff = float(d['diff'])
            context_score_dict[word] = diff
        evidence_score_dict = dict()
        for word in evidence_words:
            if word in context_score_dict:
                evidence_score_dict[word] = context_score_dict[word]
            else:
                evidence_score_dict[word] = 0
        concept_ev_dict[concept] = evidence_score_dict
    return concept_ev_dict, concept_label_dict


def count_evidence(concept_ev_dict, concept_label_dict):
    ev_concept_dict_pos = defaultdict(set)
    ev_concept_dict_neg = defaultdict(set)
    #n_ev_concepts_pos= defaultdict(set)
    #n_ev_concepts_neg= defaultdict(set)
    for concept, ev_dict in concept_ev_dict.items():
        label = concept_label_dict[concept]
        if label == 'pos':
            evidence = set()
            for context, score in ev_dict.items():
                if score > 0:
                    evidence.add(context)
                    ev_concept_dict_pos[context].add(concept)
            #n_evidence = len(evidence)
            #n_ev_concepts_pos[n_evidence].add(concept)
        elif label == 'neg':
            evidence = set()
            for context, score in ev_dict.items():
                if score > 0:
                    evidence.add(context)
                    ev_concept_dict_neg[context].add(concept)
            #n_evidence = len(evidence)
            #n_ev_concepts_neg[n_evidence].add(concept)
    return ev_concept_dict_pos, ev_concept_dict_neg
             
    

def get_prop_overview(props, model):
    prop_table = []
    for prop in props:
        concept_ev_dict, concept_label_dict = get_concept_context_matrix(prop, model)
        concepts_pos = [c for c, l in concept_label_dict.items() if l == 'pos']
        concepts_neg = [c for c, l in concept_label_dict.items() if l == 'neg']
        ev_concept_dict_pos, ev_concept_dict_neg = count_evidence(concept_ev_dict, concept_label_dict)
        total_pos = set()
        total_neg = set()
        for ev, concepts in ev_concept_dict_pos.items():
            total_pos.update(concepts)
        for ev, concepts in ev_concept_dict_neg.items():
            total_neg.update(concepts)

        prop_dict = dict()
        n_pos = len(concepts_pos)
        n_neg = len(concepts_neg)
        #print(prop, n_pos, n_neg)
        prop_dict['prop'] = prop
        prop_dict['n_ev'] = len(ev_concept_dict_pos.keys())
        prop_dict['total_pos'] = n_pos
        #prop_dict['n_ev_pos'] = len(total_pos)
        prop_dict['p_ev_pos'] = round(len(total_pos)/n_pos, 2)

        prop_dict['total_neg'] = n_neg
        #prop_dict['n_ev_neg'] = len(total_neg)
        prop_dict['p_ev_neg'] = round(len(total_neg)/n_neg, 2)
        prop_table.append(prop_dict)
    return prop_table



def get_prop_collection_overview(props, model):
    prop_collection_dict, collection_prop_dict = get_prop_types()
    collection_dict_pos = defaultdict(set)
    collection_dict_neg = defaultdict(set)
    collection_concept_cnts_pos = Counter()
    collection_concept_cnts_neg = Counter()
    collection_ev_dict = defaultdict(set)
    for prop in props:
        collection = prop_collection_dict[prop]
        concept_ev_dict, concept_label_dict = get_concept_context_matrix(prop, model)
        concepts_pos = [c for c, l in concept_label_dict.items() if l == 'pos']
        concepts_neg = [c for c, l in concept_label_dict.items() if l == 'neg']
        collection_concept_cnts_pos[collection] += len(concepts_pos)
        collection_concept_cnts_neg[collection] += len(concepts_neg)
        ev_concept_dict_pos, ev_concept_dict_neg = count_evidence(concept_ev_dict, concept_label_dict)
        # store evidence words
        collection_ev_dict[collection].update(ev_concept_dict_pos.keys())
        # collect concepts appearing with evidence words
        for ev, concepts in ev_concept_dict_pos.items():
            collection_dict_pos[collection].update(concepts)
        for ev, concepts in ev_concept_dict_neg.items():
            collection_dict_neg[collection].update(concepts)
            
    collection_table = []
    for collection, concepts_pos in collection_dict_pos.items():
        n_props = len(collection_prop_dict[collection])
        concepts_neg = collection_dict_neg[collection]
        coll_dict = dict()
        n_pos = collection_concept_cnts_pos[collection]
        n_neg = collection_concept_cnts_neg[collection]
        #print(prop, n_pos, n_neg)
        coll_dict['collection'] = collection
        coll_dict['n_props'] = n_props
        coll_dict['n_ev'] = len(collection_ev_dict[collection])
        coll_dict['total_pos'] = n_pos
        #prop_dict['n_ev_pos'] = len(total_pos)
        coll_dict['p_ev_pos'] = round(len(concepts_pos)/n_pos, 2)
        coll_dict['total_neg'] = n_neg
        #prop_dict['n_ev_neg'] = len(total_neg)
        coll_dict['p_ev_neg'] = round(len(concepts_neg)/n_neg, 2)
        collection_table.append(coll_dict)
    return collection_table


def get_prop_types():
    properties = get_properties()
    prop_collection_dict = dict()
    collection_prop_dict = defaultdict(set)
    for prop in properties:
        path = f'../data/aggregated/{prop}.json'
        with open(path) as infile:
            data = json.load(infile)
        if prop in ['warm', 'cold', 'hot']:
            collection = 'percetual-heat'
        elif prop in ['blue', 'green', 'red', 'yellow', 'black']:
            collection = 'perceptual-color'
        elif prop in ['made_of_wood']:
            collection = 'part-material'
        elif prop in ['round', 'square']:
            collection = 'perceptual-shape'
        elif prop != 'female':
            for concept, d in data.items():
                collection = d['property_type']
                break  
        else:
            collection = 'gender'
        prop_collection_dict[prop] = collection
        collection_prop_dict[collection].add(prop)
    return prop_collection_dict, collection_prop_dict



def get_relation_overview(props, model, rel_type='top', include_props = False):

    relation_table = []
    relation_concept_cnts = Counter()
    relation_ev_cnts = Counter()
    relation_property_dict = defaultdict(set)
    for prop in props:
        # get total counts to calculate proportions:
        concept_relation_dict, relation_concept_dict = get_concept_relations(prop, rel_type)
        for rel, concepts in relation_concept_dict.items():
            if prop == 'female':
                rel = 'gender-'+rel
            relation_concept_cnts[rel] += len(concepts)
        concept_ev_dict, concept_label_dict = get_concept_context_matrix(prop, model)
        ev_concept_dict_pos, ev_concept_dict_neg = count_evidence(concept_ev_dict, concept_label_dict)
        concepts_with_ev = set()
        for ev, concepts in ev_concept_dict_pos.items():
            concepts_with_ev.update(concepts)
        for ev, concepts in ev_concept_dict_neg.items():
            concepts_with_ev.update(concepts)
        for concept in concepts_with_ev:
            relations = concept_relation_dict[concept]
            for relation in relations:
                if prop == 'female':
                    relation = 'gender-'+relation
                relation_ev_cnts[relation] += 1
                relation_property_dict[relation].add(prop)
                
    for rel, ev_cnt in relation_ev_cnts.items():
        d = dict()
        d['relation'] = rel
        d['total_concepts'] = relation_concept_cnts[rel]
        d['p_evidence'] = round(ev_cnt / relation_concept_cnts[rel], 2)
        if include_props == True:
            d['properties'] = ' '.join(relation_property_dict[rel])
        relation_table.append(d)
    return relation_table


def get_concept_relations(prop, rel_type):
    #properties = get_properties()
    concept_relation_dict = dict()
    relation_concept_dict = defaultdict(set)
    path = f'../data/aggregated/{prop}.json'
    with open(path) as infile:
        data = json.load(infile)
    for concept, d in data.items():
        #print(d['relations'])
        relations_cnt = Counter(d['relations'])
        if rel_type == 'top':
            cnt_relations = defaultdict(set)
            for rel, cnt in relations_cnt.items():
                cnt_relations[cnt].add(rel)
            p = max(list(cnt_relations.keys()))
            rels = cnt_relations[p]   
            #top_rel, p = relations_cnt.most_common(1)[0]
        elif rel_type == 'hyp_top':
            rels = d['rel_hyp']
            p = d['prop_hyp']
            if prop == 'female':
                rels = [rel for rel, p in d['relations'].items() if p == 1]
                if len(rels) == 1:
                    p = 1.0
                else:
                    p = 0.0
                
        if p > 0.5:
            concept_relation_dict[concept] = rels
            for rel in rels:
                relation_concept_dict[rel].add(concept)        
    return concept_relation_dict, relation_concept_dict



def get_annotation_status(model):
    dir_annotations = f'../analysis/{model}/annotation-tfidf-top20-raw-10000'
    annotation_dict = defaultdict(dict)

    for f in os.listdir(dir_annotations):
        if not f.endswith('.csv'):
            prop = f.split('/')[-1]
            full_path = f'{dir_annotations}/{f}'
            #print(full_path)
            # get categories:
            for cat in os.listdir(full_path):
                full_path_cf = f'{full_path}/{cat}'
                annotated = [f for f in os.listdir(full_path_cf) if f.endswith('annotated.csv')]
                if len(annotated) == 1:
                    if prop in annotation_dict['complete']:
                        annotation_dict['complete'][prop].add(cat)
                    else:
                        annotation_dict['complete'][prop] = {cat}
                else:
                    if prop in annotation_dict['incomplete']:
                        annotation_dict['incomplete'][prop].add(cat)
                    else:
                        annotation_dict['incomplete'][prop] = {cat}
    return annotation_dict

def show_annotation_status(model):
    annotation_dict = get_annotation_status(model)
    # same category not annotated:
    for prop in annotation_dict['complete'].keys():
        # cats open:
        print('checking status of categories:')
        if prop in annotation_dict['incomplete']:
            print(prop)
            print('Needs annotations:')
            cats_open = annotation_dict['incomplete'][prop]
            print(cats_open)
        else:
            print(prop, 'complete')

    print()
    print('Properties not annotated yet:')
    for prop in annotation_dict['incomplete']:
        if prop not in annotation_dict['complete']:
            print(prop)
            
            
            
def check_consistency(model, prop):
    dir_annotations = f'../analysis/{model}/annotation-tfidf-top20-raw-10000'
    dir_prop = f'{dir_annotations}/{prop}-pos'
    evidence_labels = {'i', 'p', 'r', 'n', 'b', 'r'}
    categories = set()
    context_cat_dict = defaultdict(dict)
    for cat in os.listdir(dir_prop):
        categories.add(cat)
        full_path = f'{dir_prop}/{cat}/{prop}-pos-annotated.csv'
        if os.path.isfile(full_path):
            with open(full_path) as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['context']
                label  = d['evidence']
                context_cat_dict[context][cat]=label

    clean_context_cat_dict = dict()
    for context, cat_dict in context_cat_dict.items():
        annotations = cat_dict.values()
        annotations_unique = set(annotations)
        if any([l in annotations_unique for l in evidence_labels]):
            ev_label = True
        else:
            ev_label = False
        #print(len(set(annotations)))
        cats_not_present = categories.difference(set(cat_dict.keys()))
        for cat in cats_not_present:
            cat_dict[cat] = '-'
        if len(set(annotations_unique)) == 1:
            cat_dict['consistent'] = True
        else:
            cat_dict['consistent'] = False 
        cat_dict['evidence'] = ev_label
    df = pd.DataFrame(context_cat_dict).T
    return df
