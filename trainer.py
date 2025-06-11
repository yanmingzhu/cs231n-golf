from golfdb.dataloader import GolfDB, Normalize, ToTensor
from SwingNet import EventDetector
from golfdb.util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from utils import *
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

def get_lr_lambda(total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def train(model, data_loader, iterations=200, batch_size=22, seq_length=64, freeze_layer=10, lr=0.001, device=check_device()):
    print("Training a model of {} with learning rate {}".format(get_model_name(model), lr))
    print(f"iteration {iterations}, batch_size {batch_size}, seq_length {seq_length}")
    # training configuration
    split = 1
    it_save = 200  # save model every it_save iterations
    n_cpu = 6

    model.train()
    event_count = torch.zeros(9).to(device)


    criterion = torch.nn.CrossEntropyLoss(weight=get_class_weights())
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    warmup_iterations = 50
    lr_scheduler = LambdaLR(optimizer, 
                            lr_lambda=get_lr_lambda(total_steps=iterations, warmup_steps=warmup_iterations))

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    loss_list = []
    acc_list = []
    while i < iterations:
        for sample in data_loader:
            images, labels, sample_len = sample['images'].cuda(), sample['labels'].cuda(), sample['length'].cuda()
            events, count = torch.unique(labels.view(-1), return_counts=True)
            event_count[events] += count

            sample_len = sample_len.unsqueeze(1).to(device)
            seq = torch.arange(seq_length).unsqueeze(0).to(device)
            padding_masks = (seq >= sample_len).to(device)

            #todo: some don't need padding mask
            logits = model(images, padding_masks)
            labels = labels.view(batch_size*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            losses.update(loss.item(), images.size(0))
            accuracy = (logits.argmax(dim=1) == labels).sum() / len(labels)
            #print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f}), acc:{acc}'.format(i, loss=losses, acc=accuracy))
            print('Iteration: {}\t lr: {lr:.1e} Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses, lr=current_lr))

            loss_list.append(loss)
            acc_list.append(accuracy)

            #probs = F.softmax(logits.data, dim=1).cpu().numpy()
            #cred = correct_preds(probs, labels.squeeze())
            #print(f"correct {cred}")

            i += 1
            
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/{}_{}.pth.tar'.format(get_model_name(model), i))
            
            if i == iterations:
                break
    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()}, 'models/{}.pth.tar'.format(get_model_name(model)))
    return (loss_list, acc_list, event_count.to(torch.int32))
