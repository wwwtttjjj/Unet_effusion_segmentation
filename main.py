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
from functions import get_current_consistency_weight, update_ema_variables, create_model, save_model
from evaluate import evaluate
from transformers import transformer_img
import get_errormaps

if __name__ == "__main__":
    args = get_parser()

    train_path = args.train_path
    labeled_path = train_path + '/' + "labeled_data"
    weak_path = train_path + '/' + "weak_labeled_data"
    val_path = args.val_path

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
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
                                batch_size=batch_size * 2,#weak labeled data每次迭代都是batch_size的两倍
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

    '''model'''
    model = create_model(device=device,num_classes=num_classes)
    ema_model = create_model(device=device,num_classes=num_classes, ema=True)
    model.train()
    ema_model.train()

    '''some para'''
    optimizer = optim.Adam(model.parameters(),
                        lr=base_lr,
                        betas=(0.9, 0.999),
                        weight_decay=0.0001)
    consistency_criterion = losses.softmax_mse_loss
    max_epoch = max_iterations // len(weak_dataloader) + 1
    CEloss = torch.nn.CrossEntropyLoss()
    iter_num = 0
    '''wandb'''
    wandb.init(entity='wtj', project='Unet_effusion_segmentation')
    wandb.config = {'epochs': max_epoch, 'batch_size': args.batch_size}
    '''train'''
    print(f'''Starting training:
    Epochs:          {max_epoch}
    Batch size(labeled, weak_labeled):      {batch_size, batch_size * 2}
    Learning rate:   {base_lr}
    Training size:   {datasize}
    Checkpoints:     {args.save_path}
    Device:          {device.type}
    ''')
    #nclos自定义长度
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_loss = []
        division_step = 0
        for i, sampled_batch in enumerate(
                zip(cycle(labeled_dataloader), weak_dataloader)):
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
            minibatch_size = len(weak_masks)
            #噪声
            noise = torch.clamp(torch.randn_like(weak_imgs) * 0.1, -0.2, 0.2)
            #前向传播（model and emamodel）
            ema_inputs = weak_imgs + noise
            outputs_labeled = model(labeled_imgs)
            
            #计算有监督损失
            supervised_loss = CEloss(
                outputs_labeled, labeled_masks) + dice_score.dice_loss(
                    F.softmax(outputs_labeled, dim=1).float(),
                    F.one_hot(labeled_masks, num_classes).permute(0, 3, 1,
                                                                2).float(),
                    multiclass=True)
            supervised_loss = args.alpha * supervised_loss
            #一致性损失
            if args.consistency_loss:
                outputs_weak = model(weak_imgs)
                with torch.no_grad():
                    ema_outputs = ema_model(ema_inputs)
                consistency_weight = get_current_consistency_weight(args, iter_num //
                                                                    150)
                consistency_dist = consistency_criterion(outputs_weak, ema_outputs)
                # consistency_loss = consistency_weight * consistency_dist / minibatch_size
                consistency_loss = consistency_weight * consistency_dist
            else:
                consistency_loss = 0

            #计算弱监督损失
            if args.weak_loss:
                pred_labels = F.softmax(ema_outputs, dim=1).float()
                
                cf_weak_masks =  get_errormaps.get_masks(pred_labels, weak_masks, p_maps, device)
                weak_supervised_loss = CEloss(
                    outputs_weak, cf_weak_masks) + dice_score.dice_loss(
                        outputs_weak,
                        F.one_hot(cf_weak_masks, num_classes).permute(0, 3, 1,
                                                                2).float(),
                        multiclass=True)

                weak_supervised_loss = args.beta * consistency_weight * weak_supervised_loss
            else:
                weak_supervised_loss = 0
            #batch 的总共的loss
            loss = supervised_loss + weak_supervised_loss + consistency_loss


            #student model反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #teacher model EMA更新参数
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            epoch_loss.append(loss.item())
            iter_num = iter_num + 1


            wandb.log({
            'supervised_loss': supervised_loss,
            "weak_loss": weak_supervised_loss,
            "consistency_loss": consistency_loss,
            "total_loss": loss.item(),
            "step":iter_num
        })
            #评估,每个epoch评估十次
            division_step += 1
            if division_step % (len(weak_dataloader) // 10) == 0:
                val_dice = evaluate(net=model,
                    dataloader=val_dataloader,
                    device=device,
                    num_classes=num_classes)
                wandb.log({
                    "val_dice":val_dice,
                    "step":iter_num
                })
                print("\nepoch_loss : {:.3f}   Validation Dice score: {:.3f}".format(sum(epoch_loss) / len(epoch_loss), val_dice.item()))


            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1**(iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            #迭代1k保存模型
            if iter_num % 1000 == 0:
                save_model(args.save_path, iter_num, model)
            if iter_num >= max_iterations:
                break
            # break
        if iter_num >= max_iterations:
            break
        break
    save_model(args.save_path, max_iterations, model)

