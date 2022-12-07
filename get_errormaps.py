from cleanlab.filter import find_label_issues
import numpy as np
import torch


def get_masks(pred_labels, weak_masks, p_maps, device):
    labels, pred_probs, p_maps = np.array(weak_masks.detach().cpu()), np.array(
        pred_labels.detach().cpu()), np.array(p_maps)
    N, C, H, W = pred_probs.shape
    rst = []
    for batch in range(N):
        cf_pmap = p_maps[batch,:,:].reshape(H * W)
        cf_label = labels[batch,:,:].reshape(H * W)
        cf_pred = np.squeeze(pred_probs[batch,:,:,:]).reshape(-1, H * W).T
        # print(cf_label.shape, cf_pred.shape, np.unique(cf_label))
        error_maps = confinence_learning(cf_label, cf_pred)
        error_maps = update_errormaps(list(error_maps),cf_pmap)  #根据p_maps和error_maps更新error_maps
        weak_mask = label_refinement(error_maps, cf_label, cf_pred)  #label_refinement
        rst.append(weak_mask.reshape(H, W))
    weak_masks = torch.tensor(np.stack(rst)).to(device, dtype = torch.long)
    return weak_masks

    # cf_labels = labels.reshape(N * H * W, )
    # cf_pred = pred_probs.transpose(1, 0, 2, 3).reshape(-1, N * H * W).T
    # print(cf_pred.shape, cf_labels.shape)
    # error_maps = confinence_learning(cf_labels, cf_pred)
    # print(error_maps.shape)
    # error_maps = update_errormaps(list(error_maps),
    #                               p_maps)  #根据p_maps和error_maps更新error_maps
    # print(len(error_maps))
    # weak_mask = label_refinement(error_maps, cf_labels,
    #                              cf_pred)  #label_refinement
    # print(np.unique(weak_mask))

    # print(confinence_learning(labels.reshape(304200,), pred_probs.reshape(304200, 4)))
    # return weak_masks

#自信学习
def confinence_learning(labels, pred_probs):
    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,  # out-of-sample predicted probabilities from any model
        return_indices_ranked_by='self_confidence',
        min_examples_per_class = 0
    )
    return ordered_label_issues

#根据概率图谱更新错误的坐标，只取小于图谱阈值的坐标
def update_errormaps(error_maps, p_maps):
    p_maps = p_maps.flatten()
    error_maps = [i for i in range(len(error_maps)) if p_maps[i] <= 0.9]
    return error_maps

#标签修复，把自信学习后且过滤掉的错误的坐标更新为模型预测的坐标
def label_refinement(error_maps, labels, pred_probs):
    for error in error_maps:
        labels[error] = np.argmax(pred_probs[error])
    return labels

