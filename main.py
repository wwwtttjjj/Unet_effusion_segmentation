import os
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm
import wandb
import warnings
 
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from get_args import get_parser
from utils import losses, dice_score, data_loading_CLACH
from functions import get_current_consistency_weight, update_ema_variables, create_model
from evaluate import evaluate
from transformers import transformer_img
import get_errormaps

if __name__ == "__main__":
    args = get_parser()

    train_path = args.train_path
    labeled_path = train_path + '/' + "labeled_data"
    weak_path = train_path + '/' + "weak_labeled_data"
    val_path = args.val_path

    device = torch.device('cuda:' +
                        str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # batch_size = args.batch_size
    batch_size = 1
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
    '''model'''

    model = create_model(device=device,num_classes=num_classes)
    ema_model = create_model(device=device,num_classes=num_classes, ema=True)
    '''data'''

    labeled_dataset = data_loading_CLACH.BasicDataset(
        imgs_dir=labeled_path + '/' + 'imgs/',
        masks_dir=labeled_path + '/' + 'masks/',
        size=datasize,
        augumentation=transformer_img())
    weak_dataset = data_loading_CLACH.BasicDataset(
        imgs_dir=weak_path + '/' + 'imgs/',
        masks_dir=weak_path + '/' + 'masks/',
        probability_dir=weak_path + '/' + 'probability_maps/',
        size=datasize,
        augumentation=transformer_img())
    # val_dataset = data_loading_CLACH.BasicDataset(
    #     imgs_dir=val_path + '/' + 'imgs/',
    #     masks_dir=val_path + '/' + 'masks/',
    #     size=datasize,
    #     augumentation=transformer_img())
    val_dataset = data_loading_CLACH.BasicDataset(
        imgs_dir=val_path + '/' + 'imgs/',
        masks_dir=val_path + '/' + 'masks/',
        size=datasize,
        augumentation=None)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
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
    CEloss = torch.nn.CrossEntropyLoss()
    iter_num = 0
    '''wandb'''
    # wandb.init(entity='wtj', project='Unet_effusion_segmentation')
    # wandb.config = {'epochs': max_epoch, 'batch_size': args.batch_size}
    '''train'''
    #nclos自定义长度
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        loss_dict = {"supervised_loss": 0, "weak_loss": 0, "consistency_loss": 0}
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
            outputs_labeled = model(labeled_imgs)
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
            supervised_loss = 0.5 * supervised_loss
            loss = supervised_loss
            #一致性损失
            if args.consistency_loss:
                outputs_weak = model(weak_imgs)
                consistency_weight = get_current_consistency_weight(args, iter_num //
                                                                    150)
                consistency_dist = consistency_criterion(outputs_weak, ema_output)
                consistency_loss = consistency_weight * consistency_dist / minibatch_size
                loss += consistency_loss

            #计算弱监督损失
            if args.weak_loss:
                pred_labels = F.softmax(outputs_weak, dim=1).float()
                
                weak_masks =  get_errormaps.get_masks(pred_labels, weak_masks, p_maps, device)
                weak_supervised_loss = CEloss(
                    outputs_weak, weak_masks) + dice_score.dice_loss(
                        pred_labels,
                        F.one_hot(weak_masks, num_classes).permute(0, 3, 1,
                                                                2).float(),
                        multiclass=True)

                weak_supervised_loss = args.beta * consistency_weight * weak_supervised_loss
                loss += weak_supervised_loss


            #student model反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key, value in loss_dict.items():
                if key == "supervised_loss":
                    loss_dict[key] += supervised_loss.detach().cpu() / len(
                        labeled_masks)
                elif key == "weak_loss" and args.weak_loss:
                    loss_dict[key] += weak_supervised_loss.detach().cpu(
                    ) / minibatch_size
                elif key == "consistency_loss" and args.consistency_loss:
                    loss_dict[key] += consistency_loss.detach().cpu(
                    ) / minibatch_size

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
        # val_dice = evaluate(net=model,
        #                     dataloader=val_dataloader,
        #                     device=device,
        #                     num_classes=num_classes)
        total_loss = loss_dict["supervised_loss"] + loss_dict[
            "weak_loss"] + loss_dict["consistency_loss"]
        # wandb.log({
        #     'supervised_loss': loss_dict["supervised_loss"],
        #     "weak_loss": loss_dict["weak_loss"],
        #     "consistency_loss": loss_dict["consistency_loss"],
        #     "total_loss": total_loss,
        #     "val_dice": val_dice
        # })
        # print(total_loss, val_dice.item())
        print(total_loss)
        if iter_num >= max_iterations:
            break
        break
    save_mode_path = os.path.join(args.save_path,
                                'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
