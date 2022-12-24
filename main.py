import os
import random
import numpy as np
from itertools import cycle
from tqdm import tqdm
import wandb
import warnings
import logging


warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from get_args import get_parser
from utils import losses, dice_score, data_loading
from functions import get_current_consistency_weight, update_ema_variables, create_model, save_model
from evaluate import evaluate
from transformers import transformer_img, val_form, height, width
import get_errormaps

if __name__ == "__main__":
    args = get_parser()

    train_path = args.train_path
    labeled_path = train_path + '/' + "labeled_data"
    weak_path = train_path + '/' + "weak_labeled_data"
    val_path = args.val_path

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    amp = args.amp
    num_classes = 4

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    #保证可重复性
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    '''data'''
    labeled_dataset = data_loading.BasicDataset(
        imgs_dir=labeled_path + '/' + 'imgs/',
        masks_dir=labeled_path + '/' + 'masks/',
        augumentation=transformer_img())
    n_train = len(labeled_dataset)
    weak_dataset = data_loading.BasicDataset(
        imgs_dir=weak_path + '/' + 'imgs/',
        masks_dir=weak_path + '/' + 'masks/',
        probability_dir=weak_path + '/' + 'probability_maps/',
        augumentation=transformer_img())
    # val_dataset = data_loading_CLACH.BasicDataset(
    #     imgs_dir=val_path + '/' + 'imgs/',
    #     masks_dir=val_path + '/' + 'masks/',
    #     size=datasize,
    #     augumentation=transformer_img())
    val_dataset = data_loading.BasicDataset(
        imgs_dir=val_path + '/' + 'imgs/',
        masks_dir=val_path + '/' + 'masks/',
        augumentation=val_form())
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_dataloader = DataLoader(dataset=labeled_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    worker_init_fn=worker_init_fn)
    weak_dataloader = DataLoader(dataset=weak_dataset,
                                batch_size=batch_size * 2,#weak labeled data每次迭代都是batch_size的三倍
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
                                
    '''wandb'''# (Initialize logging)
    # wandb.init(entity='wtj', project='Unet_effusion_segmentation')
    # wandb.config = {'epochs': max_epoch, 'batch_size': args.batch_size}
    experiment = wandb.init(project='Unet_effusion_segmentation', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate))

    logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size(labeled, weak_labeled):      {batch_size, batch_size * 2}
    Learning rate:   {learning_rate}
    Training size:   {height, width}
    Checkpoints:     {args.save_path}
    Device:          {device.type}
    Mixed Precision: {amp}
    ''')

    '''model'''
    model = create_model(device=device,num_classes=num_classes)
    ema_model = create_model(device=device,num_classes=num_classes, ema=True)
    model.train()
    ema_model.train()

    '''some para'''
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    consistency_criterion = losses.softmax_mse_loss
    CEloss = torch.nn.CrossEntropyLoss()
    global_step = 0
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for i, sampled_batch in enumerate(
                    zip(labeled_dataloader, cycle(weak_dataloader))):
                #labeled data and weak labeded data
                labeled_imgs, labeled_masks = sampled_batch[0]['image'], sampled_batch[0]['mask']
                weak_imgs, weak_masks, p_maps = sampled_batch[1]['image'], sampled_batch[1]['mask'], sampled_batch[1]['probability_map']
                #to gpu
                labeled_imgs, labeled_masks = labeled_imgs.to(device=device, dtype=torch.float32), labeled_masks.to(
                        device=device, dtype=torch.long)
                weak_imgs, weak_masks = weak_imgs.to(device=device, dtype=torch.float32), weak_masks.to(
                        device=device, dtype=torch.long)

                minibatch_size = len(weak_masks)
                #噪声
                noise = torch.clamp(torch.randn_like(weak_imgs) * 0.1, -0.2, 0.2)
                ema_inputs = weak_imgs + noise

                #前向传播（model and emamodel）
                with torch.cuda.amp.autocast(enabled=amp):
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
                    if args.c_loss:
                        outputs_weak = model(weak_imgs)
                        with torch.no_grad():
                            ema_outputs = ema_model(ema_inputs)
                        consistency_weight = get_current_consistency_weight(args, global_step //
                                                                            150)
                        consistency_dist = consistency_criterion(outputs_weak, ema_outputs)
                        # consistency_loss = consistency_weight * consistency_dist / minibatch_size
                        consistency_loss = consistency_weight * consistency_dist
                    else:
                        consistency_loss = 0

                    #计算弱监督损失
                    if args.w_loss:
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
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)


                grad_scaler.update()
                pbar.update(labeled_imgs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                #teacher model EMA更新参数
                update_ema_variables(model, ema_model, args.ema_decay, global_step)


                experiment.log({
                'supervised_loss': supervised_loss,
                "weak_loss": weak_supervised_loss,
                "consistency_loss": consistency_loss,
                "total_loss": loss.item(),
                'epoch': epoch,
                "step":global_step
            })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #评估,每个epoch评估十次
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_P, val_S, val_I = evaluate(model, val_dataloader, device, num_classes)
                        val_dice = (val_P + val_S + val_I) / 3
                        scheduler.step(val_dice)

                        logging.info('Validation Dice score: {},{},{},{}'.format(val_P, val_S, val_I, val_dice))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation_P': val_P,
                            'validation_S': val_S,
                            'validation_I': val_I,
                            'val_score':val_dice,
                            'images': wandb.Image(labeled_imgs[0].cpu()),
                            'masks': {
                                'true': wandb.Image(labeled_masks[0].float().cpu()),
                                'pred': wandb.Image(outputs_labeled.argmax(dim=1)[0].float().cpu()),
                            },
                            "step":global_step,
                            'epoch': epoch
                        })
                #迭代1k保存模型
        if  epoch % 20 == 0:
            save_model(args.save_path, epoch, model)
    save_model(args.save_path, epoch, model)

