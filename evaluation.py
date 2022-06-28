import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import collate_fn
from sklearn.metrics import roc_auc_score
from loss import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        if key =='maha_acc':
            try:
                new_dict[key] = 0
                for i in l:
                    new_dict[key] += i[key]
            except:
                pass
        elif key =='cosine_correct':
            pass
        else:
            new_dict[key] = []
            for i in l:
                new_dict[key] += i[key]
    return new_dict


def evaluate_ood(args, model, features, ood, tag):

    keys = ['maha', 'cosine','maha_acc']

    dataloader = DataLoader(features, batch_size=args.train_batch, collate_fn=collate_fn)
    in_scores = []
    
    cosine_correct, total_len = 0, 0
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, ind = True)
            in_scores.append(ood_keys)
        cosine_correct += ood_keys['cosine_correct']
        total_len+=len(batch['labels'])

    cosine_ind_acc = float(cosine_correct/total_len)
        
    in_scores = merge_keys(in_scores, keys)
    
    dataloader = DataLoader(ood, batch_size=args.train_batch, collate_fn=collate_fn)
    out_scores = []
    out_labels_origin = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            out_scores.append(ood_keys)
            out_labels_origin.extend(batch['labels'].tolist())
    out_scores = merge_keys(out_scores, keys)
    
    outputs = {}

    for key in keys:
        if key == 'maha_acc':
            outputs[tag+"_"+key] = float(in_scores[key] /len(features))
        else:
            ins = np.array(in_scores[key], dtype=np.float64)
            outs = np.array(out_scores[key], dtype=np.float64)
            inl = np.ones_like(ins).astype(np.int64)
            outl = np.zeros_like(outs).astype(np.int64)
            scores = np.concatenate([ins, outs], axis=0)
            labels = np.concatenate([inl, outl], axis=0)

            
            auroc = get_auroc(labels, scores)
            # fpr_95 = get_fpr_95(labels, scores, return_indices=False)
            
            fpr_95= fpr_at_95(ins,outs,inl,outl)
            
            
            outputs[tag + "_" + key + "_auroc"] = auroc
            outputs[tag + "_" + key + "_fpr95"] = fpr_95
        

    outputs['cosine_acc']=cosine_ind_acc
    return outputs

def fpr_at_95(ins,outs,inl,outl):
    # calculate the falsepositive error when tpr is 95%
    
    ins = sorted(ins)
    delta = ins[len(ins)//20]

    correct_index, wrong_index=[], []
    
    for idx, out in enumerate(outs):
        if out >= delta:
            wrong_index.append(idx)
        else:
            correct_index.append(idx)

    fpr = len(wrong_index) /len(outs)

    return fpr


def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)
