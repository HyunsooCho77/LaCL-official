

import torch
import torch.nn.functional as F
import pdb
from torch import nn
import numpy as np

def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


    

def nt_xent_CCM(x, y, beta = 2, alpha = 0, t=0.2):
    # l2 norm first
    x = F.normalize(x, dim=1)
    c = [y]*len(y)
    
    mask = torch.stack(c)
    for i in range(len(mask)):
        mask[i] = mask[i]- mask[i][i]
    
    mask_pos = (mask>0).type(torch.float)  /t
    mask_neg = (mask<0).type(torch.float)  /t
    
    #! HS
    # print(mask_pos.get_device())
    # print(mask_neg.get_device())
    #! HS

    mask_eq = (mask==0).type(torch.float) * alpha

    diag = torch.eye(len(y)) * (1/t - alpha)
    diag = diag.cuda()
    

    #! HS
    # print(mask_eq.get_device())
    # print(diag.get_device())
    #! HS


    mask_sum = mask_pos + mask_neg + diag + mask_eq

    mask_sum = mask_sum.view(1,1,len(y),len(y))
    mask_final = F.interpolate(mask_sum,scale_factor = 2,mode='nearest')
    mask_final = mask_final.squeeze().cuda()

    # calculate cosine-sim between all possible pairs in minibatch
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores


    # apply mask to calculated similarity
    x_scale = x_scores * mask_final

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  
    targets[1::2] -= 1  
    
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))




def nt_xent_SPA(x,y,beta=2, alpha=0.05, t=0.2):
    # l2 norm first
    x = F.normalize(x, dim=1)

    #################attrack mask
    batch_size = len(y)
    label_info = [y]*batch_size

    att_mask = torch.stack(label_info)
    for i in range(len(att_mask)):
        att_mask[i] = att_mask[i]- att_mask[i][i]
        att_mask[i][i] = att_mask[i][i] - 1

    same_class_mask = (att_mask == 0).type(torch.uint8)
    
    label = torch.Tensor([]).long()
    for idx, row in enumerate(same_class_mask):
        random_choice = torch.nonzero(row).squeeze()
        random_idx = torch.randint(random_choice.size()[0], (1,)) 
        label = torch.cat((label,random_choice[random_idx])) 


    label = label * 2
    
    label_copy1 = torch.unsqueeze(label, 1)
    label_copy2 = torch.unsqueeze(label, 1)
    concatted = torch.cat([label_copy1, label_copy2], 1)
    result = concatted.view([-1, batch_size*2]).squeeze()
    result[1::2]+=1

    #repel mask
    c = [y]*len(y)
    mask = torch.stack(c)
    for i in range(len(mask)):
        mask[i] = mask[i]- mask[i][i]
    
    mask_pos = (mask>0).type(torch.uint8) /t
    mask_neg = (mask<0).type(torch.uint8) /t
    
    mask_rot_eq = (mask%10==0).type(torch.float) * ( beta - 1/t )
    mask_eq = (mask==0).type(torch.float) * (1/t - beta + alpha)
    
    diag = torch.eye(len(y)) * - alpha

    mask_sum = mask_pos + mask_neg + mask_rot_eq + mask_eq + diag
    
    mask_sum = mask_sum.view(1,1,len(y),len(y)).type(torch.FloatTensor)
    mask_final = F.interpolate(mask_sum,scale_factor = 2,mode='nearest')
    mask_final = mask_final.squeeze().cuda()

    for i in range(len(mask_final)):
        mask_final[i][result[i]] = 1/t

    # calculate cosine-sim between all possible pairs in minibatch
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scores * mask_final
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5
    
    return F.cross_entropy(x_scale, result.long().to(x_scale.device))


class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def _reg_loss(x1,x2, args):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    import pdb
    pdb.set_trace()
    x1_norm_t = x1_norm.transpose(dim0=0,dim1=1)

    cor_mat = torch.matmul(x1_norm_t,x2_norm).clamp(min=1e-7)
    cor_mat.div_(args.train_batch)

    on_diag = torch.diagonal(cor_mat).pow_(2).sum()
    

    # off_diag = off_diagonal(cor_mat).pow_(2).sum()
    return on_diag #+ off_diag

    # if args.reg_loss_only_identity == True:
    #     return on_diag
    # else:
    #     return on_diag + off_diag

def _reg_loss_fixed1(x1,x2, args,margin=0.5):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    cor_mat = (x1_norm.t() @ x2_norm).clamp(min=1e-7)


    diag = torch.diagonal(cor_mat)
    margin_mask = (diag>margin).type(torch.uint8)

    loss = (diag*margin_mask).sum()/args.train_batch
    
    return loss

def _reg_loss_fixed3(x1,x2, args,margin=0.6, margin2=0.3):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    cor_mat = (x1_norm.t() @ x2_norm).clamp(min=1e-7)
    diag = torch.diagonal(cor_mat)
    
    off_diag_mask = (cor_mat > margin2).type(torch.uint8).fill_diagonal_(0)
    diag_mask = (diag > 1).type(torch.uint8)

    final_mask = off_diag_mask + diag_mask * torch.eye(len(diag_mask)).cuda()

    loss = (cor_mat*final_mask).sum()/args.train_batch
    
    return loss


def _reg_loss_fixed2(x1,x2, args, margin=0.5):
    
    x1_norm = F.normalize(x1,dim = 0) # batch-wise normalization
    x2_norm = F.normalize(x2,dim = 0)
    
    cor_mat = (x1_norm @ x2_norm.t()).clamp(min=1e-7)
    
    diag = torch.diagonal(cor_mat)
    margin_mask = (diag>margin).type(torch.uint8)

    loss = (diag*margin_mask).sum()/args.train_batch
    
    return loss


def reg_loss(tok_embeddings, args):
    
    
    
    if args.layerwise_reg == False:
        
        raw_tok_embeddings, aug_tok_embeddings =  tok_embeddings
        
        
        raw_hid_first = raw_tok_embeddings[0]
        raw_hid_last = raw_tok_embeddings[-1]

        aug_hid_first = aug_tok_embeddings[0]
        aug_hid_last = aug_tok_embeddings[-1]

        # x1,x2 = tok_embeddings[0],tok_embeddings[-1]
        
        # raw_hid_1 = x1[::2,:]
        # aug_hid_1 = x1[1::2,:]

        # raw_hid_2 = x2[::2,:]
        # aug_hid_2 = x2[1::2,:]
        
        loss_reg = (_reg_loss(raw_hid_first,aug_hid_last, args) + _reg_loss(raw_hid_last,aug_hid_first, args)) / 2
        # loss_reg = (_reg_loss(raw_hid_first,raw_hid_last, args) + _reg_loss(aug_hid_first,aug_hid_last, args)) / 2
        
        return loss_reg
    
    else:
        total_reg_loss = 0
        
        raw_tok_embeddings, aug_tok_embeddings =  tok_embeddings
        # import pdb
        # pdb.set_trace()
        # for i in range(args.num_layer-1):
        for i in range(len(raw_tok_embeddings)-1):
            raw_hid = raw_tok_embeddings[i]
            raw_hid_next = raw_tok_embeddings[i+1]

            aug_hid = aug_tok_embeddings[i]
            aug_hid_next = aug_tok_embeddings[i+1]
            
            # total_reg_loss += (_reg_loss(raw_hid,aug_hid_next, args) + _reg_loss(aug_hid,raw_hid_next, args)) / 2
            
            total_reg_loss += (_reg_loss_fixed1(raw_hid,raw_hid_next, args) + _reg_loss_fixed1(aug_hid,aug_hid_next, args)) / 2
            # total_reg_loss += (_reg_loss_fixed3(raw_hid,raw_hid_next, args) + _reg_loss_fixed3(aug_hid,aug_hid_next, args)) / 2
            
        return total_reg_loss


# def reg_loss_old(tok_embeddings, args):
    
#     if args.layerwise_reg == False:

#         x1,x2 = tok_embeddings[0],tok_embeddings[-1]
        
#         raw_hid_1 = x1[::2,:]
#         aug_hid_1 = x1[1::2,:]

#         raw_hid_2 = x2[::2,:]
#         aug_hid_2 = x2[1::2,:]
        
#         loss_reg = (_reg_loss(raw_hid_1,aug_hid_2, args) + _reg_loss(raw_hid_2,aug_hid_1, args)) / 2
        
#         return loss_reg
    
#     else:
#         total_reg_loss = 0
#         for i in range(args.num_layer-1):
#             x1,x2 = tok_embeddings[i],tok_embeddings[i+1]
#             raw_hid_1 = x1[::2,:]
#             aug_hid_1 = x1[1::2,:]

#             raw_hid_2 = x2[::2,:]
#             aug_hid_2 = x2[1::2,:]
            
#             total_reg_loss += (_reg_loss(raw_hid_1,aug_hid_2, args) + _reg_loss(raw_hid_2,aug_hid_1, args)) / 2
            
#         return total_reg_loss


def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

# def nt_xent_sup(x, x_adv, mask, device, t=0.5):
def nt_xent_sup(x, x_adv, mask, device, t=0.2):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()
    dis = (x * (mask - torch.eye(x.size(0)).long().to(device)) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().to(device)) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)

    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,
                                                                     axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,
                                                               axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result


def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])
    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    print("estimated threshold:", threshold)
    return threshold

def mahalanobis_distance(x: np.ndarray,
                         y: np.ndarray,
                         covariance: np.ndarray) -> float:
    """
    Calculate the mahalanobis distance.

    Params:
        - x: the sample x, shape (num_features,)
        - y: the sample y (or the mean of the distribution), shape (num_features,)
        - covariance: the covariance of the distribution, shape (num_features, num_features)

    Returns:
        - score: the mahalanobis distance in float

    """
    num_features = x.shape[0]

    vec = x - y
    cov_inv = np.linalg.inv(covariance)
    bef_sqrt = np.matmul(np.matmul(vec.reshape(1, num_features), cov_inv), vec.reshape(num_features, 1))
    return np.sqrt(bef_sqrt).item()


def get_score(cm):
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    acc = (sum(correct) / sum(total) * 100).round(2)
    acc_in = (sum(correct[:-1]) / sum(total[:-1]) * 100).round(2)
    acc_ood = (correct[-1] / total[-1] * 100).round(2)
    # print(f"Overall(macro): , f:{f}, acc:{acc}, p:{p}, r:{r}")
    # print(f"Seen(macro): , f:{f_seen}, acc:{acc_in}, p:{p_seen}, r:{r_seen}")
    # print(f"=====> Uneen(Experiment) <=====: , f:{f_unseen}, acc:{acc_ood}, p:{p_unseen}, r:{r_unseen}")

    return f, acc, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen

def one_hot(labels, n_classes):
        one_hot = torch.zeros(labels.size(0), n_classes).cuda()
        one_hot[torch.arange(labels.size(0)), labels] = 1
        return one_hot



class ProxyAnchorLoss(nn.Module):
    def __init__(self, scale=32, margin=0.1):
        super(ProxyAnchorLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, output, label):
        pos_label = F.one_hot(label, num_classes=output.size(-1))
        neg_label = 1 - pos_label
        pos_num = torch.sum(torch.ne(pos_label.sum(dim=0), 0))
        pos_output = torch.exp(-self.scale * (output - self.margin))
        neg_output = torch.exp(self.scale * (output + self.margin))
        pos_output = (torch.where(torch.eq(pos_label, 1), pos_output, torch.zeros_like(pos_output))).sum(dim=0)
        neg_output = (torch.where(torch.eq(neg_label, 1), neg_output, torch.zeros_like(neg_output))).sum(dim=0)
        pos_loss = torch.sum(torch.log(pos_output + 1)) / pos_num
        neg_loss = torch.sum(torch.log(neg_output + 1)) / output.size(-1)
        loss = pos_loss + neg_loss
        return loss



def get_sim_mat(x):
    x = F.normalize(x, dim=1)
    return (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores

def Supervised_NT_xent(sim_matrix, labels, temperature=0.2, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    labels = labels.repeat(2)

    
     
    # labels = torch.cat([labels, labels], dim=-1).view(-1, a.shape[-1])
    
    # labels = F.interpolate(labels,scale_factor = 2,mode='nearest')

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)


    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss


def nt_xent2(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

