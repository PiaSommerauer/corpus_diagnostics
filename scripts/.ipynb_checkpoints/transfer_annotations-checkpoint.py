import os
from collections import Counter, defaultdict
import csv
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)


def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties

# transfer giga annotations to wiki

properties = get_properties()
#properties = ['dangerous']
model_name_current = 'wiki_updated'
model_name_old = 'giga_full_updated'


for prop in properties:
    print(prop)
    # current file:
    annotation_name = 'annotation-tfidf-top_3_3-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name_current}/{annotation_name}/{prop}'
    f_annotation_new = f'{path_dir_annotation}/annotation-updated.csv'
    f_annotation_tr = f'{path_dir_annotation}/annotation-transferred-updated.csv'

    # old file:
    annotation_name = 'annotation-tfidf-top_3_3-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name_old}/{annotation_name}/{prop}'
    f_annotation_old = f'{path_dir_annotation}/annotation-updated-done.csv'

    # load old annotations
    if os.path.isfile(f_annotation_old):
        print('found file')
        context_annotation_dict=dict()
        with open(f_annotation_old) as infile:
            data = list(csv.DictReader(infile))
            for d in data:
                c = d['context']
                et = d['evidence_type']
                context_annotation_dict[c] = et
                #c = d['context']

        # load new candidates

        with open(f_annotation_new) as infile:
            data = list(csv.DictReader(infile))

        # fill in old annotations
        for d in data:
            c = d['context']
            if c in context_annotation_dict:
                et = context_annotation_dict[c]
            else:
                et = 'NA'
            d['evidence_type'] = et

        # write to new file

        with open(f_annotation_tr, 'w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames = data[0].keys())
            writer.writeheader()
            for d in data:
                writer.writerow(d)