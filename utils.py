import torch
import random
import numpy as np
from torch.utils.data import Dataset
import datasets
import random
import os
import torch
import pickle

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    'clinc150': ("text", None),
    'snips': ("text", None),
    'banking77': ("text", None),
    'mix_snips': ("text", None),
    'mix_banking': ("text", None),
}


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





class PairDataset(Dataset):
    def __init__(self, raw_dset, aug_dset):
        self.raw_dset = raw_dset
        self.aug_dset = aug_dset

    def __getitem__(self, index):
        raw_data = self.raw_dset[index]
        aug_data = self.aug_dset[index]
        
        return [raw_data, aug_data]

    def __len__(self):
        return len(self.raw_dset)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if 'indices' in batch[0]:
        indices = torch.LongTensor([f["indices"] for f in batch])
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
            "indices": indices,
        }
    else:
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
        }
    return outputs


def collate_fn2(batch):
    
    batch1 = [i[0] for i in batch]
    batch2 = [i[1] for i in batch]

    max_len1 = max([len(f["input_ids"]) for f in batch1])
    max_len2 = max([len(f["input_ids"]) for f in batch2])

    max_len = max(max_len1,max_len2)

    input_ids1 = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch1]
    input_ids2 = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch2]

    input_mask1 = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch1]
    input_mask2 = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch2]

    labels1 = [f["labels"] for f in batch1]
    labels2 = [f["labels"] for f in batch2]
    
    input_ids1 = torch.tensor(input_ids1, dtype=torch.long)
    input_ids2 = torch.tensor(input_ids2, dtype=torch.long)
    input_mask1 = torch.tensor(input_mask1, dtype=torch.float)
    input_mask2 = torch.tensor(input_mask2, dtype=torch.float)
    labels1 = torch.tensor(labels1, dtype=torch.long)
    labels2 = torch.tensor(labels2, dtype=torch.long)
    
    outputs1 = {
        "input_ids": input_ids1,
        "attention_mask": input_mask1,
        "labels": labels1,
    }
    
    outputs2 = {
        "input_ids": input_ids2,
        "attention_mask": input_mask2,
        "labels": labels2,
    }
    return [outputs1, outputs2]



def write_log2(args, model_name, dic):
    if args.split:
        dir = os.path.join(args.log_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, 'eval_result')
    else:
        dir = os.path.join(args.log_dir,  f"{args.task_name}-no_split-seed{args.seed}", model_name, 'eval_result')
    
    os.makedirs(os.path.join(dir), exist_ok = True)
    log_fname = 'evaluation_log.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        for key in dic.keys():
            f.write(f'{dic[key]*100:.3f},')
        f.write('\n')



def cal_num_labels(args):
    if args.task_name == 'clinc150':
        if args.split_ratio == 0:
            num_labels = 150
        else:
            num_labels = round(15*args.split_ratio) * 10
    elif args.task_name == 'banking77':
        # assert args.split == False
        num_labels = round(77*args.split_ratio)
    elif args.task_name == 'snips':
        # assert args.split == False
        num_labels = round(7*args.split_ratio)
        # num_labels = 7
    elif args.task_name.startswith('mix'):
        mix_task_name = args.task_name[4:]
        if mix_task_name == 'snips':
            num_labels = 150 - 7
        if mix_task_name == 'banking' or  mix_task_name == 'banking77':
            mix_task_name = 'banking'
            num_labels = 150 - 30
    
    return num_labels