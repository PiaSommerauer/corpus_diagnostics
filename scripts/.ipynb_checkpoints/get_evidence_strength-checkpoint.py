import analyze_evidence

from statistics  import median

from collections import defaultdict
import os
import csv



# compare evidence strength across properties - may not make too much sense...

def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties


def get_evidence_strength_props(evidence_types, properties, model_name, top_cutoff, concept_cutoff):

    prop_et_strength_dict = defaultdict(dict)

    path_evidence = f'../analysis/{model_name}/evidence_{top_cutoff}_{concept_cutoff}-raw-10000-categories'

    for prop in properties:
        means_prop = []
        prop_dict = analyze_evidence.load_prop_data(prop)
        # only use pos examples:
        pos_ml_labels = ['all', 'all-some', 'few-some', 'some']
        concepts_pos = [c for c, d in prop_dict.items() if d['ml_label'] in pos_ml_labels]
        for et in evidence_types:
            means_et = []
            path_dir = f'{path_evidence}/{prop}/{et}'
            if os.path.isdir(path_dir):
                ev_files = [f for f in os.listdir(path_dir) if f.endswith('.csv')]
                for f in ev_files:
                    full_path = f'{path_dir}/{f}'
                    with open(full_path) as infile:
                        data = list(csv.DictReader(infile))
                        means = [float(d['mean']) for d in data if d[''] in concepts_pos]
                        if len(means) > 0:
                            mean = sum(means)/len(means)
                        else:
                            mean = 0.0
                        #means_et.append(mean)
                        means_prop.append(mean)
        if len(means_prop) > 0:
            mean_prop = sum(means_prop)/len(means_prop)
        else:
            mean_prop = 0.0
        prop_et_strength_dict[prop]['strength'] = mean_prop
        
    return prop_et_strength_dict


def get_strength_dist_props(evidence_types, properties, model_name, top_cutoff, concept_cutoff):

    prop_et_strength_dict = defaultdict(dict)

    path_evidence = f'../analysis/{model_name}/evidence_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    
    x_props = []
    y_strengths = []
    labels = []
    for prop in properties:
        means_prop = []
        prop_dict = analyze_evidence.load_prop_data(prop)
        # only use pos examples:
        pos_ml_labels = ['all', 'all-some', 'few-some', 'some']
        concepts_pos = [c for c, d in prop_dict.items() if d['ml_label'] in pos_ml_labels]
        ml_labels_pos = [d['ml_label'] for c, d in prop_dict.items() if d['ml_label'] in pos_ml_labels]
        for concept, l in zip(concepts_pos, ml_labels_pos):
            means_concept = []
            for et in evidence_types:
                path_dir = f'{path_evidence}/{prop}/{et}'
                if os.path.isdir(path_dir):
                    ev_files = [f for f in os.listdir(path_dir) if f.endswith('.csv')]
                    for f in ev_files:
                        full_path = f'{path_dir}/{f}'
                        with open(full_path) as infile:
                            data = list(csv.DictReader(infile))
                        for  d in data:
                            if d[''] == concept:
                                means_concept.append(float(d['mean']))
                      
            if len(means_concept) > 0:
                mean_concept = median(means_concept)
            else:
                mean_concept = 0.0
            x_props.append(prop)
            y_strengths.append(mean_concept)
            labels.append(l)
            
    df = pd.DataFrame(dict(prop=x_props, strength=y_strengths, label=labels))
    return df



def get_prop_strength_overview(model_name, top_cutoff, concept_cutoff):
    
    data = dict()
    
    ets = ['p', 'l', 'n']
    ets_string = '_'.join(ets)
    properties = get_properties()

    data_specific = get_evidence_strength_props(ets, properties, 
                                     model_name, top_cutoff, concept_cutoff)


    ets = ['i', 'r', 'b']
    data_non_specific = get_evidence_strength_props(ets, properties, 
                                     model_name, top_cutoff, concept_cutoff)

    for p, d in data_specific.items():

        data[p] = dict()
        data[p]['prop_specific'] = d['strength']
        d_non_sp = data_non_specific[p]
        data[p]['non_specific'] = d_non_sp['strength']
    return data
