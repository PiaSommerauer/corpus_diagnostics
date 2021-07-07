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
    label = 'pos'
    concepts_pos_total = get_concepts(model, prop, label)
    label = 'neg'
    concepts_neg_total = get_concepts(model, prop, label)

    n_pos = len(concepts_pos_total)
    n_neg = len(concepts_neg_total)
    
    print(prop, model)
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
    annotation_dict = defaultdict(set)

    for f in os.listdir(dir_annotations):
        if not f.endswith('.csv'):
            prop = f.split('/')[-1]
            full_path = f'{dir_annotations}/{f}'
            #print(full_path)
            # get categories:
            files = os.listdir(full_path)
            if 'annotation-done.csv' in files:
                annotation_dict['complete'].add(prop)
            else:
                annotation_dict['incomplete'].add(prop)
    return annotation_dict

def show_annotation_status(model):
    annotation_dict = get_annotation_status(model)
    # same category not annotated:
    print('completed:\n')
    for prop in sorted(list(annotation_dict['complete'])):
        # cats open:
        print(prop)
    print()
    print('Incomplete:\n')
    for prop in sorted(annotation_dict['incomplete']):
        if prop not in annotation_dict['complete']:
            print(prop)
            
            
def get_property_annotations(model, prop):
    dir_annotations = f'../analysis/{model}/annotation-tfidf-top20-raw-10000'
    f_ann = f'{dir_annotations}/{prop}-pos/annotation-done.csv'
    
    ann_table = dict()
    
    with open(f_ann) as infile:
        line = infile.read().split('\n')[0]
    if '\t' in line:
        delim = '\t'
    else:
        delim = ','
    #print(f_ann)
    #print(delim)
    with open(f_ann) as infile:
        data = list(csv.DictReader(infile, delimiter = delim))
    all_categories = set()
    for d in data:
        categories = d['categories'].split(' ')
        all_categories.update(categories)
    for d in data:
        d_row = dict()
        context = d['context']
        categories = d['categories'].split(' ')
        evidence = d['evidence']
        ev_found = False
        for cat in all_categories:
            if cat in categories:
                d_row[cat]  = evidence
                if evidence not in ['u', None, '-']:
                    ev_found = True
            else:
                d_row[cat] = '-'
        d_row['evidence'] = ev_found
        ann_table[context] = d_row
    df = pd.DataFrame(ann_table).T
    return df

            
            
            
def get_property_annotations_old(model, prop):
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




def get_evidence_words(prop, model):
    df = get_property_annotations(model, prop)
    evidence_words_total = dict()
    for i, row in df.iterrows():
        if row['evidence'] == True:
            row = dict(row)
            ev_values = list(set([v for v in set(row.values()) if v not in {True, False, '-'}]))
            assert len(ev_values) == 1, f'More than one label {ev_values}'
            evidence_words_total[i] = ev_values[0]
    return evidence_words_total


def get_context_dicts(path):
    context_count_dict = dict()
    with open(path) as infile:
        data = list(csv.DictReader(infile))
    for n, d in enumerate(data, 1):
        context = d['context']
        d.pop('context')
        d.pop('min_tfidf')
        d.pop('max_tfidf')
        d.pop('mean_diff')
        #d['rank'] = n
        context_count_dict[context] = d
    
    return context_count_dict


def get_concepts(model, prop, label):
    #dir_path = f'../analysis/{model}/tfidf_aggregated_concept_scores-raw-10000'
    dir_path = f'../results/{model}/tfidf-raw-10000/each_target_vs_corpus/{prop}/{label}'
    concepts = [f.split('.')[0] for f in os.listdir(dir_path)]
    return concepts
    
    
def get_concept_evidence_counts(prop, model):
    overview_table = []
    evidence_words_total = get_evidence_words(prop, model)
    # get total counts
    dir_path = f'../analysis/{model}/tfidf_aggregated_concept_scores-raw-10000'
    path_all_pos = f'{dir_path}/{prop}-pos/all-pos/{prop}-pos.csv'
    path_all_neg = f'{dir_path}/{prop}-neg.csv'
    
    data_pos = get_context_dicts(path_all_pos)
    data_neg = get_context_dicts(path_all_neg)
    
    context_count_dict = dict()
    label = 'pos'
    concepts_pos = get_concepts(model, prop, label)
    label = 'neg'
    concepts_neg = get_concepts(model, prop, label)
   
    
    for w, ev_type in evidence_words_total.items():
        d = dict()
        d_pos = data_pos[w]
        # add concept counts
        n_concepts_pos = int(d_pos['n_concepts'])
        p_concepts = round(n_concepts_pos/len(concepts_pos), 2)
        d_pos['p_concepts'] = p_concepts
        d_pos['t_concepts'] = len(concepts_pos)
        if w in data_neg:
            d_neg = data_neg[w]
            # add concept counts
            n_concepts_neg = int(d_neg['n_concepts'])
            p_concepts_neg = round(n_concepts_neg/len(concepts_neg), 2)
            d_neg['p_concepts'] = p_concepts_neg
            d_neg['t_concepts'] = len(concepts_neg)
            d_neg_new = dict()
            for k, v in d_neg.items(): 
                d_neg_new[k+'-neg'] = v
        else:
            n_concepts_neg = 0
            d_neg_new = dict()
            p_concepts_neg = 0
            for k in d_pos.keys():
                d_neg_new[k+'-neg'] = '-'
        
                
        d['word'] = w
        d['evidence_type'] = ev_type
        d['distinctiveness'] = p_concepts-p_concepts_neg
        p, r, f1 = get_f1_distinctiveness(n_concepts_pos, n_concepts_neg, 
                                          len(concepts_pos), len(concepts_neg))
        d['f1_dist'] = f1
        for k, v in d_pos.items():
            d[k+'-pos'] = v
        for k, v in d_neg_new.items():
            d[k] = v
        overview_table.append(d)
    return overview_table 



def get_concept_evidence(path_concepts, evidence_words_total):
    concept_context_dict = defaultdict(set)
    context_concept_dict= defaultdict(set)
    total_concepts = set()
    for f in os.listdir(path_concepts):
        if  f.endswith('.csv'):
            concept = f.split('.')[0]
            total_concepts.add(concept)
            with open(f'{path_concepts}/{f}') as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['']
                diff = float(d['diff'])
                if diff > 0:
                    if context in evidence_words_total.keys():
                        concept_context_dict[concept].add(context)
                        context_concept_dict[context].add(concept)
    return concept_context_dict, context_concept_dict, total_concepts
    
    
def get_f1_distinctiveness(n_pos, n_neg, total_pos, total_neg):
    
    tp = n_pos
    tn = total_neg - n_neg
    fp = n_neg
    fn = total_pos - n_pos
    
    if tp+fp != 0:
        p = tp/(tp+fp)
    else:
        p = 0
    if tp+fn != 0:
        r = tp/(tp+fn)
    else:
        r = 0
    
    if p+r != 0:
        f1 = 2 * ((p*r)/(p+r))
    else:
        f1=0
    
    return p, r, f1
    

def get_concept_context_overview(prop, model):

    evidence_words_total = get_evidence_words(prop, model)
    #print(evidence_words_total)

    concept_dir = f'../results/{model}/tfidf-raw-10000/each_target_vs_corpus'
    path_pos = f'{concept_dir}/{prop}/pos'
    path_neg = f'{concept_dir}/{prop}/neg'
    
    c_e_pos, e_c_pos, t_pos = get_concept_evidence(path_pos, evidence_words_total)
    c_e_neg, e_c_neg, t_neg = get_concept_evidence(path_neg, evidence_words_total)
    #print(len(c_e_pos), len(c_e_neg))
    
 
    table = []
    evidence_type = ['all', 'p', 'n', 'i', 'r', 'b', 'l']
    
    for et in evidence_type:
        if et == 'all':
            ev_words = evidence_words_total.keys()
        else:
            ev_words = [w for w, ev_type in 
                        evidence_words_total.items() if ev_type == et]
        #concept_ev_dict_pos = defaultdict(set)
        #concept_ev_dict_neg = defaultdict(set)
        concepts_et_pos = set()
        concepts_et_neg = set()
        for ew in ev_words:
            # concepts with evidence words
            concepts_pos = e_c_pos[ew]
            concepts_neg = e_c_neg[ew]
            concepts_et_pos.update(concepts_pos)
            concepts_et_neg.update(concepts_neg)
        n_pos = len(concepts_et_pos)
        n_neg = len(concepts_et_neg)
        p_pos = n_pos/len(t_pos)
        p_neg = n_neg/len(t_neg)
        d = dict()
        p, r, f1 = get_f1_distinctiveness(n_pos, n_neg, len(t_pos), len(t_neg))
        d['evidence'] = et
        d['distinctiveness'] = p_pos-p_neg
        d['f1_dist'] = f1
        d['n_concepts_with_ev_pos'] = n_pos
        d['p_concepts_with_ev_pos'] = p_pos
        d['n_concepts_pos'] = len(t_pos)
        d['n_concepts_with_ev_neg'] = n_pos
        d['p_concepts_with_ev_neg'] = p_neg
        d['n_concepts_neg'] = len(t_neg)
        d['total_evidence_words'] = len(ev_words)
        table.append(d)
    
    return table


    




#### updated version

def aggregate_old_annotations(prop, model):
    dir_annotations = f'../analysis/{model}/annotation-tfidf-top20-raw-10000'
    dir_prop = f'{dir_annotations}/{prop}-pos'
    context_cat_dict = defaultdict(set)
    context_evidence_dict = dict()
    for cat in os.listdir(dir_prop):
        full_path = f'{dir_prop}/{cat}/{prop}-pos-annotated.csv'
        if os.path.isfile(full_path):
            with open(full_path) as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['context']
                evidence = d['evidence']
                context_cat_dict[context].add(cat)
                context_evidence_dict[context] = evidence
    path_for_annotation = f'{dir_prop}/annotation-done.csv'
    with open(path_for_annotation, 'w') as outfile:
        outfile.write('context,evidence,categories\n')
        for context, categories in context_cat_dict.items():
            evidence = context_evidence_dict[context]
            outfile.write(f'{context},{evidence},{" ".join(categories)}\n')
            
            
def summarize_annotation_table(prop, model):
    dir_annotations = f'../analysis/{model}/annotation-tfidf-top20-raw-10000'
    dir_prop = f'{dir_annotations}/{prop}-pos'
    context_cat_dict = defaultdict(set)
    for cat in os.listdir(dir_prop):
        if cat not in  ['annotation.csv',  'annotation-done.csv']:
            full_path = f'{dir_prop}/{cat}/{prop}-pos.csv'
            with open(full_path) as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['context']
                context_cat_dict[context].add(cat)
    path_for_annotation = f'{dir_prop}/annotation.csv'
    with open(path_for_annotation, 'w') as outfile:
        outfile.write('context,evidence,categories\n')
        for context, categories in context_cat_dict.items():
            outfile.write(f'{context}, ,{" ".join(categories)}\n')
            
            
            
def get_prop_overview(model, prop):
    prop_dict = dict()
    prop_dict['property'] = prop

    evidence_words = get_evidence_words(prop, model)
    prop_dict['n_evidence_words'] = len(evidence_words)
    evidence_type = [ 'p', 'n', 'i', 'r', 'b', 'l']
    for e in evidence_type:
        prop_dict[e] = 0
    for ew, t in evidence_words.items():
        prop_dict[t] += 1

    # get combined distinctiveness
    df_combined = pd.DataFrame(get_concept_context_overview(prop, model))
    for i, row in df_combined.iterrows():
        et = row['evidence']
        if et == 'all':
            prop_dict['combined_dist'] = row['distinctiveness']
            prop_dict['combined_f1'] = row['f1_dist']

    # get max distinctiveness
    df_overview = pd.DataFrame(get_concept_evidence_counts(prop, model))
    df_overview = df_overview.sort_values('distinctiveness', ascending=False)
    for i, row in df_overview.iterrows():
        prop_dict['max_dist'] = row['distinctiveness']
        prop_dict['max_dist_f1'] = row['f1_dist']
        prop_dict['max_dist_ev'] = row['word']
        prop_dict['max_dist_t'] = row['evidence_type']
        break
    return prop_dict
