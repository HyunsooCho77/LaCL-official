import numpy as np
import torch
from torch import nn, optim

def tpr95(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the falsepositive error when tpr is 95%
    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)

    start = min(ind)
    end = max(ind)
    gap = (end- start)/gap

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(ind >= delta)) / np.float(len(ind))
        fprTemp = np.sum(np.sum(ood >= delta)) / np.float(len(ood))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += fprTemp
            total += 1
    fpr = fpr/total

    return fpr


def auroc(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the AUROC
    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)
 
    start = min(ind)
    end = max(ind)
    gap = (end- start)/gap

    auroc = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(ind >= delta)) / np.float(len(ind))
        fpr = np.sum(np.sum(ood > delta)) / np.float(len(ood))
        auroc += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    auroc += fpr * tpr

    return auroc

def auprIn(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the AUPR
    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)
 
    start = min(ind)
    end = max(ind)
    gap = (end- start)/gap

    auprIn = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(ind >= delta)) / np.float(len(ind))
        fp = np.sum(np.sum(ood >= delta)) / np.float(len(ood))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        auprIn += (recallTemp-recall)*precision
        recallTemp = recall
    auprIn += recall * precision

    return auprIn

def auprOut(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the AUPR
    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)
 
    start = min(ind)
    end = max(ind)
    gap = (end- start)/gap

    auprOut = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(ind < delta)) / np.float(len(ind))
        tp = np.sum(np.sum(ood < delta)) / np.float(len(ood))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprOut += (recallTemp-recall)*precision
        recallTemp = recall
    auprOut += recall * precision
    
    return auprOut


def metric(ind_fname = None, ood_fname = None, delta = None, gap = 20000):
    if ind_fname == None or ood_fname == None:
        fpr = tpr95()
        aur = auroc()
        auprin = auprIn()
        auprout = auprOut()
    else :
        fpr = tpr95(ind_fname,ood_fname, gap)
        aur = auroc(ind_fname,ood_fname, gap)
        auprin = auprIn(ind_fname,ood_fname, gap)
        auprout = auprOut(ind_fname,ood_fname, gap)

    print(f'fpr at tpr95 : {fpr*100:2f}')
    print(f'AUROC : {aur*100:2f}')
    print(f'AUPR ind : {auprin*100:2f}')
    print(f'AUPR ood : {auprout*100:2f}')

    return fpr, aur, auprin, auprout


def threshold_min(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the falsepositive error when tpr is 95%
    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)

    delta = min(ind)
    print(delta)

    new_ind = ind[ind>= delta]
    new_ood = ood[ood>= delta]
    print(len(new_ind)*100/len(ind))
    print(len(new_ood)*100/len(ood))
    tpr = np.sum(np.sum(ind >= delta)) / np.float(len(ind))
    print(tpr*100)
    fpr = np.sum(np.sum(ood >= delta)) / np.float(len(ood))
    print(fpr*100)



def get_threshold(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):

    ind = np.loadtxt(ind_fname)
    ood = np.loadtxt(ood_fname)

    ind = np.sort(ind)
    ood = np.sort(ood)

    max_idx = int(len(ind) * 0.05)
    min_idx = 0

    prev_delta = ind[min_idx]
    prev_tpr = ind[ind>=prev_delta]
    prev_fpr = ood[ood>=prev_delta]
    
    tpr = len(prev_tpr)*100/len(ind)
    fpr = len(prev_fpr)*100/len(ood)
    
    
    probe_range = 15
    total = 0
    prev_flag = False
    for value in np.arange(min_idx+probe_range, max_idx, probe_range):
        total +=1
        
        delta = ind[value]
        
        new_ind = ind[ind>=delta]
        new_ood = ood[ood>=delta]

        next_tpr = len(new_ind)*100/len(ind)
        next_fpr = len(new_ood)*100/len(ood)

        ratio = (fpr- next_fpr) / (tpr- next_tpr)
        if ratio <= 1 :
            if prev_flag == True:
                print(ratio)
                print(total)
                print(f'anomaly in fpr, tpr at :{next_fpr:.3f}, {next_tpr:.3f}')
                break
            else:
                prev_flag = True
        else:
            prev_flag = False


        tpr = next_tpr
        fpr = next_fpr




def get_delta(ind_fname='./softmax_scores/confidence_Base_In.txt', ood_fname='./softmax_scores/confidence_Base_Out.txt', gap = 20000):
    # calculate the falsepositive error when tpr is 95%
    ind = np.loadtxt(ind_fname)

    start = min(ind)
    end = max(ind)
    gap = (end- start)/gap

    total = 0.0
    fpr = 0.0
    delta_sum = 0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(ind >= delta)) / np.float(len(ind))
        if tpr <= 0.9505 and tpr >= 0.9495:
            delta_sum += delta
            total += 1
    delta = delta_sum/total
    return delta

def sm_calculator(f_name, d_loader, embeddings, model, ind = True):
    f = open(f_name, 'w')

    model.eval()
    for data in d_loader:
        inputs, length = data.text
        labels = data.label

        inputs = embeddings(inputs)
        outputs = model(inputs = inputs, length= list(length))

        sm = nn.Softmax(dim=1)
        sm_outputs = sm(outputs)
        sm_outputs, preds = torch.max(sm_outputs, dim =1 )
        
        if ind == False:
            for output in sm_outputs:
                f.write("{}\n".format(output.detach().cpu().numpy()))
        else :
            for idx, output in enumerate(sm_outputs):
                if labels[idx] == preds[idx]:
                    f.write("{}\n".format(output.detach().cpu().numpy()))

import os

if __name__ == '__main__':
    
    
    fname_ind = os.path.join('trained_models/oecc','ind_temp.txt')
    fname_ood = os.path.join('trained_models/oecc','ood_temp.txt')
    total = 94004

    get_threshold(fname_ind,fname_ood)
    # threshold_min(fname_ind,fname_ood)