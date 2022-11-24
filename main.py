import argparse
import os
import time
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from unet import UNet
from utils import data_loading, ramps, losses, dice_score
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--train_path',
                    type=str,
                    default='./data/train',
                    help='the path of training data')
parser.add_argument('--val_path',
                    type=str,
                    default='./data/val',
                    help='the path of val data')
parser.add_argument('--save_path',
                    type=str,
                    default='./checkpoints',
                    help='the path of save_model')
parser.add_argument('--deterministic',
                    type=int,
                    default=0,
                    help='whether use deterministic training')
parser.add_argument('--max_iterations',
                    type=int,
                    default=6000,
                    help='maximum epoch number to train')

parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--base_lr',
                    type=float,
                    default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help='the batch_size of training size')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency',
                    type=float,
                    default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup',
                    type=float,
                    default=40.0,
                    help='consistency_rampup')
args = parser.parse_args()

train_path = args.train_path
labeled_path = train_path + '/' + "labeled_data"
weak_path = train_path + '/' + "weak_labeled_data"
val_path = args.val_path

device = torch.device('cuda:' +
                      str(args.gpu) if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
num_classes = 4
datasize = (400, 400)

#保证可重复性
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,
                                                   args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):

    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False):
    # Network definition
    net = UNet(n_channels=3, n_classes=num_classes)
    model = net.cuda()
    #截断反向传播的梯度流
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


model = create_model()
ema_model = create_model(ema=True)
'''data'''


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


labeled_dataset = data_loading.BasicDataset(
    imgs_dir=labeled_path + '/' + 'imgs/',
    masks_dir=labeled_path + '/' + 'masks/',
    size=datasize)
weak_dataset = data_loading.BasicDataset(imgs_dir=weak_path + '/' + 'imgs/',
                                         masks_dir=weak_path + '/' + 'masks/',
                                         probability_dir=weak_path + '/' +
                                         'probability_maps/',
                                         size=datasize)
labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
weak_dataloader = DataLoader(dataset=weak_dataset,
                             batch_size=batch_size * 2,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)
val_dataset = data_loading.BasicDataset(imgs_dir=val_path + '/' + 'imgs/',
                                        masks_dir=val_path + '/' + 'masks/',
                                        size=datasize)
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
'''some para'''
model.train()
ema_model.train()
optimizer = optim.SGD(model.parameters(),
                      lr=base_lr,
                      momentum=0.9,
                      weight_decay=0.0001)
consistency_criterion = losses.softmax_mse_loss
max_epoch = max_iterations // len(weak_dataloader) + 1
CEloss = torch.nn.CrossEntropyLoss(reduction='mean')
iter_num = 0
'''train'''
#nclos自定义长度
print('start training')
for epoch_num in tqdm(range(max_epoch), ncols=70):
    for i, sampled_batch in enumerate(
            zip(labeled_dataloader, cycle(weak_dataloader))):
        #labeled data and weak labeded data
        labeled_imgs, labeled_masks = sampled_batch[0]['image'], sampled_batch[
            0]['mask']
        weak_imgs, weak_masks, p_maps = sampled_batch[1][
            'image'], sampled_batch[1]['mask'], sampled_batch[1][
                'probability_map']
        #to gpu
        labeled_imgs, labeled_masks = labeled_imgs.to(
            device=device, dtype=torch.float32), labeled_masks.to(
                device=device, dtype=torch.long).squeeze(dim=1)
        weak_imgs, weak_masks = weak_imgs.to(
            device=device, dtype=torch.float32), weak_masks.to(
                device=device, dtype=torch.long).squeeze(dim=1)
        #噪声
        noise = torch.clamp(torch.randn_like(weak_imgs) * 0.1, -0.2, 0.2)
        #前向传播（model and emamodel）
        ema_inputs = weak_imgs + noise
        outputs_labeled, outputs_weak = model(labeled_imgs), model(weak_imgs)
        minibatch_size = len(weak_masks)
        with torch.no_grad():
            ema_output = ema_model(ema_inputs)

        #计算有监督损失
        supervised_loss = CEloss(
            outputs_labeled, labeled_masks) + dice_score.dice_loss(
                F.softmax(outputs_labeled, dim=1).float(),
                F.one_hot(labeled_masks, num_classes).permute(0, 3, 1,
                                                              2).float(),
                multiclass=True)

        weak_supervised_loss = CEloss(
            outputs_weak, weak_masks) + dice_score.dice_loss(
                F.softmax(outputs_weak, dim=1).float(),
                F.one_hot(weak_masks, num_classes).permute(0, 3, 1, 2).float(),
                multiclass=True)
        # loss_seg = F.cross_entropy(outputs, labeled_masks)
        # outputs_soft = F.softmax(outputs, dim=1)
        #一致性损失
        consistency_weight = get_current_consistency_weight(iter_num // 150)
        consistency_dist = consistency_criterion(outputs_weak, ema_output)
        consistency_loss = consistency_weight * consistency_dist / minibatch_size

        weak_supervised_loss = 5 * consistency_weight * weak_supervised_loss
        loss = supervised_loss + weak_supervised_loss + consistency_loss

        #student model反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #teacher model EMA更新参数
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        iter_num = iter_num + 1

        ## change lr
        if iter_num % 2500 == 0:
            lr_ = base_lr * 0.1**(iter_num // 2500)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        #迭代1k保存模型
        if iter_num % 1000 == 0:
            save_mode_path = os.path.join(args.save_path,
                                          'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
        if iter_num >= max_iterations:
            break
    val_dice = evaluate(net=model,
                        dataloader=val_dataloader,
                        device=device,
                        num_classes=num_classes)
    print(val_dice)
    if iter_num >= max_iterations:
        break

save_mode_path = os.path.join(args.save_path,
                              'iter_' + str(max_iterations) + '.pth')
torch.save(model.state_dict(), save_mode_path)
