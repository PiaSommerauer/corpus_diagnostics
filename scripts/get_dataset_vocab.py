import os
import json

def load_data(prop):
    path = '../data/aggregated/'+prop+'.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict


def create_vocab(props):

    vocab = set()
    for prop in props:
        prop_dict = load_data(prop)
        target_pos = [k for k, d in prop_dict.items() if d['ml_label'] in ['all', 'all-some', 'few-some']]
        target_neg = [k for k, d in prop_dict.items() if d['ml_label'] in ['few']]
        vocab.update(target_pos)
        vocab.update(target_neg)
    return vocab


def vocab_to_file(vocab):
    with open('../data/vocab.txt', 'w') as outfile:
        for w in vocab:
            outfile.write(w+'\n')


def main():
    props = ['female',
             'yellow', 'red', 'green', 'blue', 'black',
             'cold', 'hot', 'warm',
             'round', 'square',
             'swim', 'fly', 'roll', 'lay_eggs',
             'wings', 'wheels', 'made_of_wood',
             'juicy', 'sweet', 'used_in_cooking', 'dangerous']

    vocab = create_vocab(props)
    vocab_to_file(vocab)

if __name__ == '__main__':
    main()

