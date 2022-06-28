import enum
import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader
import pdb
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt
import pickle
import os
from utils import collate_fn, collate_fn_ref1, log_pred_results
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from loss import *
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def evaluate_acc(args, model, eval_dataset, tag="train"):
    dataloader = DataLoader(eval_dataset, batch_size=100, collate_fn=collate_fn)
    label_total, label_correct = 0, 0
    eval_bar = tqdm(dataloader)
    for step, batch in enumerate(eval_bar):
        model.eval()
        labels = batch["labels"].cuda()
        batch = {key: value.to(DEVICE) for key, value in batch.items()}

        batch["labels"] = None
        batch["indices"] = None
        outputs = model(**batch)
        
        sm = nn.Softmax(dim=1)
        sm_outputs = sm(outputs['logits'])
        sm_outputs, preds = torch.max(sm_outputs, dim=1)

        label_total += len(labels)
        # find correct intent labels
        label_correct += (labels == preds).float().sum()
        ind_acc = label_correct*100/label_total
        eval_bar.set_description(f'Test {tag} accuracy : {ind_acc:.3f}')
    return ind_acc

def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        if key =='maha_acc' or key == 'layerwise_maha_sum_acc':
            try:
                new_dict[key] = 0
                for i in l:
                    new_dict[key] += i[key]
            except:
                pass
        elif key =='layerwise_maha':
            new_dict[key] = []
            for layer in range(13):
                temp = []
                for i in l:
                    temp += i[key][layer]
                new_dict[key].append(temp)
                    
            pass
        elif key =='layerwise_maha_acc':
            new_dict[key] = [0] * 12
            for layer in range(12):
                total = 0
                for i in l:
                    try:
                        total +=i[key][layer]
                    except:
                        pass # ood dataset일 경우
                new_dict[key][layer] = total
        elif key =='cosine_correct':
            pass
        else:
            new_dict[key] = []
            for i in l:
                new_dict[key] += i[key]
    return new_dict


def evaluate_ood(args, model, features, ood, tag, final_rep):
    if args.loss_ce == True:
        keys = ['softmax', 'maha', 'cosine', 'energy','maha_acc']
    else:
        keys = ['maha', 'cosine','maha_acc']

    dataloader = DataLoader(features, batch_size=args.batch_size, collate_fn=collate_fn)
    in_scores = []
    
    cosine_correct, total_len = 0, 0
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, ind = True, final_rep= final_rep)
            in_scores.append(ood_keys)
        cosine_correct += ood_keys['cosine_correct']
        total_len+=len(batch['labels'])

    cosine_ind_acc = float(cosine_correct/total_len)
        
    in_scores = merge_keys(in_scores, keys)


    
    dataloader = DataLoader(ood, batch_size=args.batch_size, collate_fn=collate_fn)
    out_scores = []
    out_labels_origin = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch,final_rep=final_rep)
            out_scores.append(ood_keys)
            out_labels_origin.extend(batch['labels'].tolist())
    out_scores = merge_keys(out_scores, keys)
    
    # get mahalanobis distance of in-domain for histogram
    # if args.split:
    #     hist_dir = os.path.join(args.save_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model.model_name, 'maha_histogram')
    # else:
    #     hist_dir = os.path.join(args.save_dir, f"{args.task_name}-no_split-seed{args.seed}", model.model_name, 'maha_histogram')
    # os.makedirs(hist_dir, exist_ok=True)
    
    # maha_list_ind = in_scores['maha']
    # with open(os.path.join(hist_dir, "ind_maha_scores.txt"), "w") as f:
    #     [f.write(f'{maha}\n') for maha in maha_list_ind]
    # maha_list_ood = out_scores['maha']
    # with open(os.path.join(hist_dir, "ood_maha_scores.txt"), "w") as f:
    #     [f.write(f'{maha}\n') for maha in maha_list_ood]
    
    # hist_max = 0
    # hist_min = min(min(maha_list_ind), min(maha_list_ood))
    # bins = int((hist_max - hist_min) // 40)
    # plt.hist(maha_list_ind, bins, label='IND', histtype='step', density=True)
    # plt.hist(maha_list_ood, bins, label='OOD', histtype='step', density=True)
    # plt.legend()
    # plt.savefig(os.path.join(hist_dir, 'maha_histogram.png'))
    # plt.close('all')
    
    outputs = {}
    a,b,c,d = 0,0,0,0
    cosine_s, maha_s, maha_l = [], [], []

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
            
            # fpr_95,wrong_indices = get_fpr_95(labels, scores, return_indices=True)
            fpr_90, correct_index_90, wrong_index_90 = HS_fpr_90(ins,outs,inl,outl)
            fpr_95, correct_index_95, wrong_index_95 = HS_fpr_95(ins,outs,inl,outl)
            
            #!#!#!#!#!#! MAHA cosine
            if key == 'maha':
                a = correct_index_90
                b = wrong_index_90
                c = correct_index_95
                d = wrong_index_95
                maha_s = scores
                maha_l = labels
            elif key == 'cosine':
                cosine_s = scores
            
            outputs[tag + "_" + key + "_auroc"] = auroc

            outputs[tag + "_" + key + "_fpr90"] = fpr_90
            outputs[tag + "_" + key + "_fpr95"] = fpr_95
        

    outputs['cosine_acc']=cosine_ind_acc
    return outputs, a, b, c, d, maha_s, cosine_s,  maha_l

def HS_fpr_95(ins,outs,inl,outl):
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

    return fpr, correct_index, wrong_index


def HS_fpr_90(ins,outs,inl,outl):
    # calculate the falsepositive error when tpr is 95%
    
    ins = sorted(ins)
    delta = ins[len(ins)//10]

    correct_index, wrong_index=[], []
    
    for idx, out in enumerate(outs):
        if out >= delta:
            wrong_index.append(idx)
        else:
            correct_index.append(idx)

    fpr = len(wrong_index) /len(outs)

    return fpr, correct_index, wrong_index



    





def evaluate_layerwise_ood(args, model, features, ood, tag):
    keys = ['layerwise_maha','layerwise_maha_acc','layerwise_maha_sum_acc']

    dataloader = DataLoader(features, batch_size=args.batch_size, collate_fn=collate_fn)
    in_scores = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.layerwise_compute_ood(**batch, ind = True)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)

    dataloader = DataLoader(ood, batch_size=args.batch_size, collate_fn=collate_fn)
    out_scores = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.layerwise_compute_ood(**batch)
            out_scores.append(ood_keys)
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:
        
        if key =='layerwise_maha':
            ins_whole = []
            outs_whole = []
            for i in range(1+12):
                ins = np.array(in_scores[key][i], dtype=np.float64)
                outs = np.array(out_scores[key][i], dtype=np.float64)
                
                ins_whole.append(ins)
                outs_whole.append(outs)

                inl = np.ones_like(ins).astype(np.int64)
                outl = np.zeros_like(outs).astype(np.int64)
                scores = np.concatenate([ins, outs], axis=0)
                labels = np.concatenate([inl, outl], axis=0)

                auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

                outputs[tag + "_" + key + f"layer{i}_auroc"] = auroc
                outputs[tag + "_" + key + f"layer{i}_fpr95"] = fpr_95
            
            # ins_sum = sum(ins_whole)
            # outs_sum = sum(outs_whole)
            # sum_inl = np.ones_like(ins).astype(np.int64)
            # sum_outl = np.zeros_like(outs).astype(np.int64)
            # scores = np.concatenate([ins_sum, outs_sum], axis=0)
            # labels = np.concatenate([sum_inl, sum_outl], axis=0)

            # auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

            # outputs[tag + "_" + key + f"layer_sum_auroc"] = auroc
            # outputs[tag + "_" + key + f"layer_sum_fpr95"] = fpr_95

        elif key =='layerwise_maha_acc':
            for i in range(12):
                acc = float(in_scores[key][i]/ len(features))
                outputs[tag+"_"+key+f'layer{i}'] = acc
        elif key == 'layerwise_maha_sum_acc':
            outputs[tag+"_"+key] = float(in_scores[key] /len(features))

            
    return outputs

def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction, recall_level=0.95,return_indices=False):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1

    if return_indices:
        score, indices = fpr_and_fdr_at_recall(new_key, prediction, recall_level=recall_level, return_indices=True)
        return score, indices
    else:
        score = fpr_and_fdr_at_recall(new_key, prediction, recall_level=recall_level, return_indices=False)
        return score

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, return_indices, recall_level=0.95, pos_label=1.):
    
    # 1,0 label -> True, False label
    y_true = (y_true == pos_label)

    # y_score -> ranking으로 변환
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    
    # re-indexing (?)
    y_score = y_score[desc_score_indices]
    
    # re-indexing (?)    
    y_true = y_true[desc_score_indices]
    
    y_wrong_indices = desc_score_indices[np.where(y_true == False)[0]]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]

    fps = 1 + threshold_idxs - tps


    recall = tps / tps[-1]


    last_ind = tps.searchsorted(tps[-1])


    sl = slice(last_ind, None, -1) # last ind부터 역순으로

    # recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    
    
    recall, fps = np.r_[recall[sl], 1], np.r_[fps[sl], 0]


    cutoff = np.argmin(np.abs(recall - recall_level))

    if return_indices:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), y_wrong_indices
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def evaluate_ood_ref1(args, model, features, ood, tag, tokenizer): # ood score of entire dataset
    # features: test, ind
    # ood: test, ood
    keys = ['raw_probs', 'raw_ht', 'softmax', 'maha', 'cosine', 'energy', 'maha_acc']
    f1_keys = ['msp', 'gda', 'lof']
    dataloader = DataLoader(features, batch_size=args.batch_size, collate_fn=collate_fn_ref1)
    in_scores = []
    in_labels = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, ind = True)
            in_scores.append(ood_keys)
            in_labels.extend(batch['labels'].tolist())
    in_scores = merge_keys(in_scores, keys)
    

    dataloader = DataLoader(ood, batch_size=args.batch_size, collate_fn=collate_fn_ref1)
    out_scores = []
    out_labels = []
    out_labels_origin = []
    ood_label = max(in_labels) + 1
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            out_scores.append(ood_keys)
            out_labels.extend([ood_label] * len(batch['labels']))
            out_labels_origin.extend(batch['labels'].tolist())
    out_scores = merge_keys(out_scores, keys)
    
    
    # get mahalanobis distance of in-domain for histogram
    if args.split:
        hist_dir = os.path.join(args.save_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model.model_name, 'maha_histogram')
    else:
        hist_dir = os.path.join(args.save_dir, f"{args.task_name}-no_split-seed{args.seed}", model.model_name, 'maha_histogram')
    os.makedirs(hist_dir, exist_ok=True)
    
    maha_list_ind = in_scores['maha']
    with open(os.path.join(hist_dir, "ind_maha_scores.txt"), "w") as f:
        [f.write(f'{maha}\n') for maha in maha_list_ind]
    maha_list_ood = out_scores['maha']
    with open(os.path.join(hist_dir, "ood_maha_scores.txt"), "w") as f:
        [f.write(f'{maha}\n') for maha in maha_list_ood]
        
    hist_max = 0
    hist_min = min(min(maha_list_ind), min(maha_list_ood))
    bins = int((hist_max - hist_min) // 30)
    plt.hist(maha_list_ind, bins, label='IND', histtype='step', density=True)
    plt.hist(maha_list_ood, bins, label='OOD', histtype='step', density=True)
    plt.legend()
    plt.savefig(os.path.join(hist_dir, 'maha_histogram.png'))
    
    outputs = {}
    scores, labels = {}, {}
    for key in keys:
        print(f'Evaluate ood, {key}')
        if key != 'maha_acc':
            ins = np.array(in_scores[key], dtype=np.float64)
            outs = np.array(out_scores[key], dtype=np.float64)
            inl = np.ones_like(ins).astype(np.int64)
            outl = np.zeros_like(outs).astype(np.int64)
            scores[key] = np.concatenate([ins, outs], axis=0)
            labels[key] = np.concatenate([inl, outl], axis=0)
            if key in ['raw_probs', 'raw_ht']:
                continue
            if key != 'maha':
                auroc, fpr_95 = get_auroc(labels[key], scores[key]), get_fpr_95(labels[key], scores[key])
            else: # get error case from fpr95
                auroc = get_auroc(labels[key], scores[key])
                fpr_95, fpr_indices = get_fpr_95(labels[key], scores[key], return_indices=True)
                total_error_cnt = int(fpr_95 * len(outs))
                
                if args.task_name == 'clinc150':
                    with open(os.path.join('data', 'clinc150', 'intentLabel2names.json')) as f:
                        label2name_origin = json.load(f)
                    label2name = {}
                    if args.split:
                        with open(os.path.join('data', 'clinc150', f'fromours_ratio_{args.split_ratio}_raw2split.pkl'), 'rb') as f:
                            split_pkl = pickle.load(f, encoding='utf-8')
                        for k, v in split_pkl.items():
                            label2name[v] = label2name_origin[str(k)][0]
                    else:
                        for k, v in label2name_origin.items():
                            label2name[int(k)] = label2name_origin[k][0]
                    for k, v in split_pkl.items():
                        label2name[v] = label2name_origin[str(k)][0]
                else:
                    with open(os.path.join('data', args.task_name, f'labels_{args.split_ratio}.json')) as f:
                        name2label = json.load(f)
                    label2name = {}
                    for k, v in name2label.items():
                        label2name[v] = k
                
                if args.split:
                    error_dir = os.path.join(
                        args.save_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model.model_name, f'Wrong_OOD({key})')
                else:
                    error_dir = os.path.join(
                        args.save_dir, f"{args.task_name}-no_split-seed{args.seed}", model.model_name, f'Wrong_OOD({key})')
                os.makedirs(error_dir, exist_ok=True)
                
                with open(os.path.join(error_dir, "error_cases.tsv"), "w") as f:
                    csv_writer = csv.writer(f, delimiter='\t')
                    title = ['Index', 'Text', 'True_label', 'Pred_label', 'OOD_score']
                    csv_writer.writerow(title)

                    for fpr_i in fpr_indices[:total_error_cnt]:
                        score = scores[key][fpr_i]
                        pdb.set_trace()
                        input_ids = torch.LongTensor(ood[fpr_i - len(inl)]['input_ids']).unsqueeze(0).to(model.device)
                        attention_mask = ood[fpr_i - len(inl)]['attention_mask'].type(torch.FloatTensor).unsqueeze(0).to(model.device)
                        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            
                        sm = nn.Softmax(dim=1)
                        sm_outputs = sm(model_output[0])
                        _, preds = torch.max(sm_outputs, dim=1)
                        
                        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ood[fpr_i - len(inl)]['input_ids'], skip_special_tokens=True))
                        text = TreebankWordTokenizer().tokenize(text)
                        text = TreebankWordDetokenizer().detokenize(text)
                        
                        true_label = label2name[out_labels_origin[fpr_i - len(inl)]]
                        pred_label = label2name[preds.item()]

                        csv_writer.writerow([fpr_i, text, true_label, pred_label, score])

            outputs[tag + "_" + key + "_auroc"] = auroc
            outputs[tag + "_" + key + "_fpr95"] = fpr_95
        else:
            outputs[tag + "_" + key] = float(in_scores[key] / len(features))
    in_labels = np.array(in_labels).astype(np.int64)
    out_labels = np.array(out_labels).astype(np.int64)
    test_labels = np.concatenate([in_labels, out_labels], axis=0)
    classes = model.all_classes + [ood_label]
    print(f'classes: {classes}')
    for ood_method in f1_keys:
        if args.split:
            pred_dir = os.path.join(
                args.save_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model.model_name)
        else:
            pred_dir = os.path.join(
                args.save_dir, f"{args.task_name}-no_split-seed{args.seed}", model.model_name)
        
        # if args.use_feature:
        #     pred_dir += f'-feature_gda'
        pred_dir = os.path.join(pred_dir, ood_method)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "results.txt"), "w") as f_out:
            f_out.write('')
        with open(os.path.join(pred_dir, "results.json"), "w") as f_out:
            f_out.write('')
        # compute gda, lof, msp here
        print("OOD method:", ood_method)
        if ood_method == 'msp':
            ins = np.array(in_scores['raw_probs'], dtype=np.float64)
            outs = np.array(out_scores['raw_probs'], dtype=np.float64)
            scores_ood = np.concatenate([ins, outs], axis=0)
            seen_conf = ins.max(axis=1) * -1.0
            unseen_conf = outs.max(axis=1) * -1.0
            threshold = -1.0 * estimate_best_threshold(seen_conf, unseen_conf)
            
            y_pred_score = scores_ood.max(axis=1)
            y_pred = scores_ood.argmax(axis=1)
            for i, score in enumerate(y_pred_score):
                if score < threshold:
                    y_pred[i] = ood_label
            auroc, fpr_95 = get_auroc(labels['softmax'], y_pred_score), get_fpr_95(labels['softmax'], y_pred_score)
            outputs[tag + "_" + ood_method + "_auroc"] = auroc
            outputs[tag + "_" + ood_method + "_fpr95"] = fpr_95
            
        
        elif ood_method == 'lof':
            y_pred_lof = pd.Series(model.lof.predict(scores['raw_ht']))
            
            y_pred = scores['raw_probs'].argmax(axis=1)
            y_pred[y_pred_lof[y_pred_lof == -1].index] = ood_label
            threshold = 0
            auroc = 0
            fpr_95 = 0
        elif ood_method == 'gda': # gda_lsqr_auto_mahalanobis
            # ood, ind 반대 방향
            distance_type = 'mahalanobis'
            gda = model.gda
            
            if args.use_feature:
                ins = np.array(in_scores['raw_ht'], dtype=np.float64)
                outs = np.array(out_scores['raw_ht'], dtype=np.float64)
                scores_ood = np.concatenate([ins, outs], axis=0)
            else:
                ins = np.array(in_scores['raw_probs'], dtype=np.float64)
                outs = np.array(out_scores['raw_probs'], dtype=np.float64)
                scores_ood = np.concatenate([ins, outs], axis=0)
            gda_result = confidence(scores_ood, gda.means_, distance_type, gda.covariance_)
            y_pred_score = gda_result.min(axis=1) # maha_score = -maha_score, should make this reversed
            auroc, fpr_95 = get_auroc(labels['softmax'], np.reciprocal(y_pred_score)), get_fpr_95(labels['softmax'], np.reciprocal(y_pred_score))
            outputs[tag + "_" + ood_method + "_auroc"] = auroc
            outputs[tag + "_" + ood_method + "_fpr95"] = fpr_95
            
            # additional_thresholds = list(range(0, 80, 2))
            # # additional_thresholds = [x/2 for x in additional_thresholds]
            # for thres in additional_thresholds:
            #     y_pred = gda.predict(scores['raw_probs'])
            #     for i, score in enumerate(y_pred_score):
            #         if score > thres:
            #             y_pred[i] = ood_label
            #     cm = confusion_matrix(test_labels, y_pred, labels=classes)
            #     f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
            #     log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen,
            #                         classes, pred_dir, cm, auroc, fpr_95, thres)
            
            # just check the best result on test set to get threshold
            seen_m_dist = confidence(ins, gda.means_, distance_type, gda.covariance_).min(axis=1)
            unseen_m_dist = confidence(
                outs, gda.means_, distance_type, gda.covariance_).min(axis=1)
            threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)
            y_pred = gda.predict(scores['raw_ht'])
            
            # if args.use_feature:
            #     y_pred = gda.predict(scores['raw_ht'])
            # else:
            #     y_pred = gda.predict(scores['raw_probs'])
            for i, score in enumerate(y_pred_score):
                if score > threshold:
                    y_pred[i] = ood_label
                
            
        cm = confusion_matrix(test_labels, y_pred, labels=classes)
        f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
        log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen,
                            classes, pred_dir, cm, auroc, fpr_95, threshold)
        
        
    return outputs





def evaluate(args, model, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        batch["labels"] = None
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results




def evaluate3(args, model, eval_dataset, tokenizer, tag="train"):
    if args.task_name == 'clinc150':
        with open(os.path.join(hydra.utils.to_absolute_path('data'), 'clinc150', 'intentLabel2names.json')) as f:
            label2name_origin = json.load(f)
        with open(os.path.join(hydra.utils.to_absolute_path('data'), 'clinc150', f'fromours_ratio_{args.split_ratio}_raw2split.pkl'), 'rb') as f:
            split_pkl = pickle.load(f, encoding='utf-8')
        label2name = {}
        for k, v in split_pkl.items():
            label2name[v] = label2name_origin[str(k)][0]
    else:
        with open(os.path.join(hydra.utils.to_absolute_path('data'), args.task_name, f'labels_{args.split_ratio}.json')) as f:
            name2label = json.load(f)
        label2name = {}
        for k, v in name2label.items():
            label2name[v] = k
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    label_total, label_correct = 0, 0
    eval_bar = tqdm(dataloader)
    if args.split:
        error_dir = os.path.join(args.save_dir, args.augment_method, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model.model_name, 'ind')
    else:
        error_dir = os.path.join(args.save_dir, args.augment_method, f"{args.task_name}-no_split-seed{args.seed}", model.model_name, 'ind')
    os.makedirs(error_dir, exist_ok=True)
    
    with open(os.path.join(error_dir, "error_cases.tsv"), "w") as f:
        csv_writer = csv.writer(f, delimiter='\t')
        title = ['Index', 'Text', 'True_label', 'Pred_label']
        csv_writer.writerow(title)
        for step, batch in enumerate(eval_bar):
            model.eval()
            labels = batch["labels"].cuda()
            indices = batch["indices"].to(args.device)
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            batch["labels"] = None
            outputs = model(**batch)
            
            sm = nn.Softmax(dim=1)
            sm_outputs = sm(outputs[0])
            sm_outputs, preds = torch.max(sm_outputs, dim=1)

            # labels = labels.squeeze(1)
            label_total += len(labels)
            # find correct intent labels
            label_correct += (labels == preds).float().sum()
            indices = indices[labels != preds].tolist()
            
            
            if len(indices) > 0:
                wrong_loc = (labels != preds)
                labels = labels[wrong_loc].tolist()
                preds = preds[wrong_loc].tolist()
                input_ids = batch['input_ids'][wrong_loc]
                for i, input_id, true_label, pred_label in zip(indices, input_ids, labels, preds):
                    
                    input_string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=True))
                    input_string = TreebankWordTokenizer().tokenize(input_string)
                    input_string = TreebankWordDetokenizer().detokenize(input_string)
                    
                    true_label = label2name[true_label]
                    pred_label = label2name[pred_label]
                    
                    csv_writer.writerow([i, input_string, true_label, pred_label])
            ind_acc = label_correct*100/label_total
            eval_bar.set_description(f'Test {tag} accuracy : {ind_acc:.3f}')
    return ind_acc
