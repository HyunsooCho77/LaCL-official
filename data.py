import datasets
from datasets import load_dataset
import random
import os
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import pickle
import pdb

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    '20ng': ("text", None),
    'trec': ("text", None),
    'imdb': ("text", None),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
    'clinc150': ("text", None),
    'snips': ("text", None),
    'banking77': ("text", None),
    'mix_snips': ("text", None),
    'mix_banking': ("text", None),
}



    # 'clinc150':150,
    # 'clinc150_0.25':40,
    # 'clinc150_0.5':80,
    # 'clinc150_0.75':110

def load(task_name, tokenizer, max_seq_length=512, split=False, ratio=0.25):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    if task_name in ('mnli', 'rte'):
        datasets = load_glue(task_name)
    elif task_name == 'sst2':
        datasets = load_sst2()
    elif task_name == '20ng':
        datasets = load_20ng()
    elif task_name == 'trec':
        datasets = load_trec()
    elif task_name == 'imdb':
        datasets = load_imdb()
    elif task_name == 'wmt16':
        datasets = load_wmt16()
    elif task_name == 'multi30k':
        datasets = load_multi30k()
    elif task_name == 'clinc150':
        datasets = load_clinc150(split=split, ratio=ratio)
    elif task_name == 'banking77':
        datasets = load_banking77(split=split, ratio=ratio)
    elif task_name == 'snips':
        datasets = load_snips(split=split, ratio=ratio)


    def preprocess_function(example_tuple):
        idx, examples = example_tuple
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        result['indices'] = idx
        return result
    task_name_prefix = task_name.split('_')[0]
    if task_name_prefix in ['clinc150', 'banking77', 'snips']:
        train_dataset = list(map(preprocess_function, enumerate(datasets['train']))) if 'train' in datasets else None
        dev_dataset = list(map(preprocess_function, enumerate(datasets['validation']))) if 'validation' in datasets else None
        test_ind_dataset = list(map(preprocess_function, enumerate(datasets['test_ind']))) if 'test_ind' in datasets else None
        test_ood_dataset = list(map(preprocess_function, enumerate(datasets['test_ood']))) if 'test_ood' in datasets else None
        return train_dataset, dev_dataset, test_ind_dataset, test_ood_dataset
    else:
        sys.exit('Wrong dataset name!')
        train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets else None
        dev_dataset = list(map(preprocess_function, datasets['validation'])) if 'validation' in datasets else None
        test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
        return train_dataset, dev_dataset, test_dataset


def load_glue(task):
    datasets = load_dataset("glue", task)
    if task == 'mnli':
        test_dataset = [d for d in datasets['test_matched']] + [d for d in datasets['test_mismatched']]
        datasets['test'] = test_dataset
    return datasets


def load_20ng():
    all_subsets = ('18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware', '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space', '18828_soc.religion.christian', '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc', '18828_talk.religion.misc')
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        dataset = load_dataset('newsgroup', subset)['train']
        examples = [{'text': d['text'], 'label': i} for d in dataset]
        random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        num_dev = int(0.1 * len(examples))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train: num_train + num_dev]
        test_dataset += examples[num_train + num_dev:]
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}

    # print(train_dataset[:5])
    # print(test_dataset[:5])
    return datasets


def load_trec():
    datasets = load_dataset('trec')
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in idxs[-num_reserve:]]
    train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in idxs[:-num_reserve]]
    test_dataset = [{'text': d['text'], 'label': d['label-coarse']} for d in test_dataset]
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_imdb():
    datasets = load_dataset('imdb')
    train_dataset = datasets['train']
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in idxs[-num_reserve:]]
    train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in idxs[:-num_reserve]]
    test_dataset = datasets['test']
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_wmt16():
    datasets = load_dataset('wmt16', 'de-en')
    test_dataset = [d['translation'] for d in datasets['test']]
    datasets = {'test': test_dataset}
    return datasets


def load_multi30k():
    test_dataset = []
    for file_name in ('./data/multi30k/test_2016_flickr.en', './data/multi30k/test_2017_mscoco.en', './data/multi30k/test_2018_flickr.en'):
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    example = {'text': line, 'label': 0}
                    test_dataset.append(example)
    datasets = {'test': test_dataset}
    return datasets


def load_sst2():
    def process(file_name):
        examples = []
        with open(file_name, 'r') as fh:
            for line in fh:
                splits = line.split()
                label = splits[0]
                text = " ".join(splits[1:])
                examples.append(
                    {'sentence': text, 'label': int(label)}
                )
        return examples
    datasets = load_dataset('glue', 'sst2')
    train_dataset = datasets['train']
    dev_dataset = datasets['validation']
    test_dataset = process('./data/sst2/test.data')
    # print(train_dataset[:5])
    # print(test_dataset[:5])
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets



def load_split_dict(dset_name, ratio):
    assert str(ratio) not in [0.25,0.5,0.75]
    with open (f'./data/{dset_name}/fromours_ratio_{ratio}_raw2split.pkl','rb') as  f:
        split_label_dict = pickle.load(f)
    
    return split_label_dict

def load_clinc150(split=False, ratio=0.25):
    root_dir = './data/clinc150'
    
    source_path = os.path.join(root_dir, 'data_full.json')
    with open(source_path, encoding='utf-8') as f:
        docs = json.load(f)

    intents_val = []  # Every intents are in each dataset split.
    [intents_val.append(intent) for _, intent in docs['val']
        if intent not in intents_val]
    # add out-of-scope intent
    intents_val.append('oos')



    # domain classification
    # this file can be found on official github page of original clinc paper
    domain_path = os.path.join(root_dir, 'domains.json')
    with open(domain_path, encoding='utf-8') as f:
        domain_docs = json.load(f)
    # add out-of-scope intent
    domain_docs['oos'] = ['oos']

    intent2domain = {}
    intentLabel2names = {}

    for i, (domain, intents) in enumerate(domain_docs.items()):
        for intent in intents:
            intent_label = intents_val.index(intent)
            intent2domain[intent_label] = i
            intentLabel2names[intent_label] = (intent, domain)
    label2names_path = os.path.join(root_dir, 'intentLabel2names.json')
    with open(label2names_path, 'w') as f:
        json.dump(intentLabel2names, f)

    
    train_dataset = []
    dev_dataset = []
    test_ind_dataset = []
    test_ood_dataset = []

    if split == False:
        for mode in docs.keys():    
        #     is_augment = 'train' in mode and 'oos' not in mode

            for i, line in enumerate(tqdm(docs[mode], desc=f'{mode} set')):
                text, intent = line
                intent_label = int (intents_val.index(intent))
                
                example = {'text': text, 'label': intent_label}

                # print(example)
                if mode == 'train':
                    train_dataset.append(example)
                elif mode == 'val':
                    dev_dataset.append(example)
                elif mode == 'test':
                    test_ind_dataset.append(example)
                elif mode == 'oos_test':
                    test_ood_dataset.append(example)

        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
        return datasets
    else:
        label_dict = load_split_dict('clinc150',ratio)
        n_ind_classes = round(15 * ratio) * 10 -1
        for mode in ['train','val','test']:    
        #     is_augment = 'train' in mode and 'oos' not in mode

            for i, line in enumerate(tqdm(docs[mode], desc=f'{mode} set')):
                text, intent = line
                intent_label = int (intents_val.index(intent))
                split_label = label_dict[intent_label]
                example = {'text': text, 'label': split_label}

                # print(example)
                if mode == 'train':
                    if label_dict[intent_label] <= n_ind_classes:
                        train_dataset.append(example)

                elif mode == 'val':
                    if label_dict[intent_label] <= n_ind_classes:
                        dev_dataset.append(example)
                elif mode == 'test':
                    if label_dict[intent_label] <= n_ind_classes:
                        test_ind_dataset.append(example)
                    else :
                        test_ood_dataset.append(example)


        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
        return datasets

def load_banking77(split=False, ratio=0.25):
    category_path = os.path.join('data', 'banking77', 'categories.json')
    with open(category_path, 'r') as f:
        categories = json.load(f)
    
    datasets = {}
    test_ood_datasets = []

    if split == False:
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join('data', 'banking77', mode)
            label_path = os.path.join(data_path, 'label')
            text_path = os.path.join(data_path, 'seq.in')
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                labels = [label.rstrip('\n') for label in labels]
            
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
                texts = [text.rstrip('\n') for text in texts]
                
            examples = []
            for text, label in zip(texts, labels):
                label_num = categories.index(label)
                examples.append({'text': text, 'label': label_num})
        
            datasets[mode] = examples
    else:
        label_dict = load_split_dict('banking77',ratio)
        n_ind_classes = round(77 * ratio) -1
        train_dataset = []
        dev_dataset = []
        test_ind_dataset = []
        test_ood_dataset = []
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join('data', 'banking77', mode)
            label_path = os.path.join(data_path, 'label')
            text_path = os.path.join(data_path, 'seq.in')
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                labels = [label.rstrip('\n') for label in labels]
            
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
                texts = [text.rstrip('\n') for text in texts]
                
            examples = []
            for text, label in zip(texts, labels):
                label_num = categories.index(label)
                examples.append({'text': text, 'label': label_num})
        
            datasets[mode] = examples

            if mode == 'train':
                for text, label in zip(texts, labels):
                    if label_dict[categories.index(label)] <= n_ind_classes:
                        example = {'text': text, 'label': label_dict[categories.index(label)]}
                        train_dataset.append(example)
            elif mode == 'valid':
                for text, label in zip(texts, labels):
                    if label_dict[categories.index(label)] <= n_ind_classes:
                        example = {'text': text, 'label': label_dict[categories.index(label)]}
                        dev_dataset.append(example)
            elif mode == 'test':
                for text, label in zip(texts, labels):
                    example = {'text': text, 'label': label_dict[categories.index(label)]}
                    if label_dict[categories.index(label)] <= n_ind_classes:
                        test_ind_dataset.append(example)
                    else:
                        test_ood_dataset.append(example)
                        
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    return datasets


def load_snips(split=False, ratio=0.25):
    # category_path = os.path.join('data', 'banking77', 'categories.json')
    # with open(category_path, 'r') as f:
    categories = {}
    
    datasets = {}
    test_ood_datasets = []
    
    if split == False:
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join('data', 'snips', mode)
            label_path = os.path.join(data_path, 'label')
            text_path = os.path.join(data_path, 'seq.in')
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                labels = [label.rstrip('\n') for label in labels]
            if mode == 'train': # if first iteration
                label_idx = 0
                for label in labels:
                    if label not in categories.keys():
                        categories[label] = label_idx
                        label_idx += 1
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
                texts = [text.rstrip('\n') for text in texts]
                
            examples = []
            for text, label in zip(texts, labels):
                examples.append({'text': text, 'label': categories[label]})
        
            datasets[mode] = examples
    else:
        label_dict = load_split_dict('snips',ratio)
        n_ind_classes = round(7 * ratio) -1
        train_dataset = []
        dev_dataset = []
        test_ind_dataset = []
        test_ood_dataset = []

        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join('data', 'snips', mode)
            label_path = os.path.join(data_path, 'label')
            text_path = os.path.join(data_path, 'seq.in')
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                labels = [label.rstrip('\n') for label in labels]
            if mode == 'train': # if first iteration
                label_idx = 0
                for label in labels:
                    if label not in categories.keys():
                        categories[label] = label_idx
                        label_idx += 1
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
                texts = [text.rstrip('\n') for text in texts]

            if mode == 'train':
                for text, label in zip(texts, labels):
                    if label_dict[categories[label]] <= n_ind_classes:
                        example = {'text': text, 'label': label_dict[categories[label]]}
                        train_dataset.append(example)
            elif mode == 'valid':
                for text, label in zip(texts, labels):
                    if label_dict[categories[label]] <= n_ind_classes:
                        example = {'text': text, 'label': label_dict[categories[label]]}
                        dev_dataset.append(example)
            elif mode == 'test':
                for text, label in zip(texts, labels):
                    example = {'text': text, 'label': label_dict[categories[label]]}
                    if label_dict[categories[label]] <= n_ind_classes:
                        test_ind_dataset.append(example)
                    else:
                        test_ood_dataset.append(example)

        print(train_dataset[:5])
        print(dev_dataset[:5])
        print(test_ind_dataset[:5])
        print(test_ood_dataset[:5])
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    return datasets



import random
from random import shuffle

def split_pickle_file(n_classes):

    for seed, ratio in enumerate([0.25,0.5,0.75]):
        random.seed(seed)
        shuffled_l = list(range(n_classes))
        shuffle(shuffled_l)
        
        shuffle_dic = {}
        for idx, raw_index in enumerate(shuffled_l):
            shuffle_dic[raw_index]= idx

        with open(f'ratio_{ratio}_raw2split.pkl','wb') as f:
            pickle.dump(shuffle_dic, f)
        print(shuffle_dic)


if __name__ == "__main__":
    split_pickle_file(77)



import pickle
def LACL_load(task_name, tokenizer, max_seq_length=512, split=False, ratio=0.25):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    
    if ratio == 0:
        with open(f'./data/clinc150/raw_dset_whole.pkl','rb') as f:
            dataset_raw = pickle.load(f)
        with open(f'./data/clinc150/aug_dset_whole.pkl','rb') as f:
            dataset_aug = pickle.load(f)
    else:
        with open(f'./data/{task_name}/raw_dset_ratio{ratio}.pkl','rb') as f:
            dataset_raw = pickle.load(f)
        with open(f'./data/{task_name}/aug_dset_ratio{ratio}.pkl','rb') as f:
            dataset_aug = pickle.load(f)


    def preprocess_function(example_tuple):
        idx, examples = example_tuple
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        result['indices'] = idx
        return result
    

    train_raw_dataset = list(map(preprocess_function, enumerate(dataset_raw['train']))) if 'train' in dataset_raw else None
    train_aug_dataset = list(map(preprocess_function, enumerate(dataset_aug['train']))) if 'train' in dataset_aug else None
    test_ind_dataset = list(map(preprocess_function, enumerate(dataset_raw['test_ind']))) if 'test_ind' in dataset_raw else None
    test_ood_dataset = list(map(preprocess_function, enumerate(dataset_raw['test_ood']))) if 'test_ood' in dataset_raw else None


    l_ind = []

    l_train = []
    for dat in test_ind_dataset:
        l_ind.append(dat['labels'])

    for dat in train_raw_dataset:
        l_train.append(dat['labels'])

    print(list(set(l_ind)))
    print(list(set(l_train)))


    return train_raw_dataset, train_aug_dataset, test_ind_dataset, test_ood_dataset


# augment_method -> augment에 augment
# augment_both -> raw에 augment
def LACL_load_aug(args, tokenizer):
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    print("Loading {}".format(args.task_name))

    if 'mix' in args.task_name:
        mix_task_name = args.task_name[4:]
        print(f'MIX SETTING, mixed task: {mix_task_name}')
        if mix_task_name.startswith('banking'):
            mix_task_name = 'banking'
        if 'bt' in args.augment_method:
            with open(f'data/clinc150/aug_dset_clinc4{mix_task_name}.pkl','rb') as f:
                dataset_aug = pickle.load(f)
            with open(f'data/clinc150/raw_dset_clinc4{mix_task_name}.pkl','rb') as f:
                dataset_raw = pickle.load(f)
        else:
            with open(f'data/clinc150/raw_dset_clinc4{mix_task_name}.pkl','rb') as f:
                dataset_raw = pickle.load(f)
            with open(f'data/clinc150/raw_dset_clinc4{mix_task_name}.pkl','rb') as f:
                dataset_aug = pickle.load(f)

    elif args.split_ratio == 0:
        if 'bt' in args.augment_method:
            with open(f'./data/clinc150/aug_dset_whole.pkl','rb') as f:
                dataset_aug = pickle.load(f)
            if 'bt' in args.augment_both:
                with open(f'./data/clinc150/aug_dset_whole_additional.pkl','rb') as f:
                    dataset_raw = pickle.load(f)
            else:
                with open(f'./data/clinc150/raw_dset_whole.pkl','rb') as f:
                    dataset_raw = pickle.load(f)
        else:
            with open(f'./data/clinc150/raw_dset_whole.pkl','rb') as f:
                dataset_raw = dataset_aug = pickle.load(f)


    else:
        if 'bt' in args.augment_method:
            with open(f'./data/{args.task_name}/aug_dset_ratio{args.split_ratio}.pkl','rb') as f:
                dataset_aug = pickle.load(f)
            if 'bt' in args.augment_both:
                with open(f'./data/{args.task_name}/aug_dset_ratio{args.split_ratio}_additional.pkl','rb') as f:
                    dataset_raw = pickle.load(f)
            else:
                with open(f'./data/{args.task_name}/raw_dset_ratio{args.split_ratio}.pkl','rb') as f:
                    dataset_raw = pickle.load(f)
        else:
            with open(f'./data/{args.task_name}/raw_dset_ratio{args.split_ratio}.pkl','rb') as f:
                dataset_raw = dataset_aug = pickle.load(f)


    def preprocess_function(example_tuple):
        idx, examples = example_tuple
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=args.max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        result['indices'] = idx
        return result
    
    def preprocess_function_aug(example_tuple):
        idx, examples = example_tuple
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        if 'span' in args.augment_method:
            if len(inputs[0]) <= args.span_mask_len or len(inputs[0]) < 30: # if too short, no span masking
                pass
            else:
                ind = np.random.randint(len(inputs[0]) - args.span_mask_len)
                left, right = inputs[0].split(inputs[0][ind:ind + args.span_mask_len], 1)
                inputs = (" ".join([left, tokenizer.mask_token, right]), )
        if 'mlm' in args.augment_method:
            ...
        result = tokenizer(*inputs, max_length=args.max_seq_length, truncation=True)
        if 'shuffle' in args.augment_method:
            np.random.shuffle(result['input_ids'])
        if 'token_cutoff' in args.augment_method:
            cutoff_size = int(len(result['input_ids']) * args.cutoff_ratio)
            for i in range(cutoff_size):
                cutoff_idx = np.random.randint(0, len(result['input_ids']))
                for v in result.values():
                    v.pop(cutoff_idx)
        result["labels"] = examples["label"] if 'label' in examples else 0
        result['indices'] = idx
        return result
    
    # train-input1
    train_aug1_dataset = list(map(preprocess_function_aug, enumerate(dataset_raw['train']))) if args.augment_both != 'None' \
                        else list(map(preprocess_function, enumerate(dataset_raw['train'])))
    # train-input2
    train_aug2_dataset = list(map(preprocess_function_aug, enumerate(dataset_aug['train'])))
    
    # test ind
    test_ind_dataset = list(map(preprocess_function, enumerate(dataset_raw['test_ind']))) if 'test_ind' in dataset_raw else \
                        list(map(preprocess_function, enumerate(dataset_aug['test_ind'])))
    # test ood
    test_ood_dataset = list(map(preprocess_function, enumerate(dataset_raw['test_ood']))) if 'test_ood' in dataset_raw else \
                        list(map(preprocess_function, enumerate(dataset_aug['test_ood'])))

    # train-raw
    train_raw_dataset = list(map(preprocess_function, enumerate(dataset_raw['train'])))


    l_ind = []

    l_train = []
    for dat in test_ind_dataset:
        l_ind.append(dat['labels'])

    for dat in train_raw_dataset:
        l_train.append(dat['labels'])

    print(list(set(l_ind)))
    print(list(set(l_train)))

    return train_aug1_dataset, train_aug2_dataset, test_ind_dataset, test_ood_dataset, train_raw_dataset


def load_ref1(task_name, tokenizer, max_seq_length=512, split=False, ratio=0.25):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    
    if not split:
        assert task_name=='clinc150'
        with open(f'./data/{task_name}/raw_dset_whole.pkl','rb') as f:
            dataset_raw = pickle.load(f)
        # train_dataset, _, test_ind_dataset, test_ood_dataset = load(task_name, tokenizer, max_seq_length, split)
        # for dataset in [train_dataset, test_ind_dataset, test_ood_dataset]:
        #     for i, data_dict in enumerate(dataset):
        #         for k, v in data_dict.items():
        #             if k != 'labels':
        #                 dataset[i][k] = torch.LongTensor(v).unsqueeze(0)
        # pdb.set_trace()
    else:
        with open(f'./data/{task_name}/raw_dset_ratio{ratio}.pkl','rb') as f:
            dataset_raw = pickle.load(f)

    def preprocess_function(example_tuple):
        idx, examples = example_tuple
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True, return_tensors='pt')
        result["labels"] = examples["label"] if 'label' in examples else 0
        result['indices'] = idx
        return result
    def apply_preprocess(data):
        return list(map(preprocess_function, enumerate(data)))
    train_dataset = apply_preprocess(dataset_raw['train']) if 'train' in dataset_raw else None
    test_ind_dataset = apply_preprocess(dataset_raw['test_ind']) if 'test_ind' in dataset_raw else None
    test_ood_dataset = apply_preprocess(dataset_raw['test_ood']) if 'test_ood' in dataset_raw else None
    l_ind = []
    l_train = []
    for dat in test_ind_dataset:
        l_ind.append(dat['labels'])

    for dat in train_dataset:
        l_train.append(dat['labels'])
    print(list(set(l_ind)))
    print(list(set(l_train)))
    return train_dataset, test_ind_dataset, test_ood_dataset


def write_aug_data(args, ):
    if not args.write_aug_data:
        return
    
    # if args.augment_method == 'bt':
        
    # elif args.augment_method == 'ss':
        
    else:
        ...