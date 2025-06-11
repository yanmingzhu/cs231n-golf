import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from utils import *
from detector_factory import get_model
from dataloader_factory import get_test_data_loader, get_val_data_loader
from golfdb.util import correct_preds

def eval(model, dataloader, split, seq_length, n_cpu, disp):
    correct = []
    delta_list = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        '''
        pred = probs.argmax(axis=1)
        pred_idx = np.where(pred != 8)[0]
        pred_val = pred[pred_idx]
        #print(f"pred_idx = {pred_idx}, pred_val = {pred_val}")
        pred_dic = {int(idx):int(pred_val[i]) for i, idx in enumerate(pred_idx)}
        '''
        l1 = labels.squeeze()
        label_dic = [i for i, c in enumerate(labels.squeeze()) if c != 8]

        events = np.where(labels < 8)[0]
        preds = np.zeros(len(events))

        for j in range(len(events)):
            preds[j] = np.argsort(probs[:, j])[-1]
        
        #print(f"preds {preds} --- labels {label_dic}")

        events, preds, deltas, tol, c = correct_preds(probs, labels.squeeze())
        '''
        print(f"events {events}")
        print(f"preds {preds}" )
        print(f"deltas {deltas}, tol {tol}")
        print(f"correct {np.where(deltas <= tol)}")
        '''
        if disp:
            print(i, c, tol)
        correct.append(c)
        '''
        print(f"deltas {deltas}, tol {tol}")
        print(f"deltas-tol {deltas - tol}")
        print(f"deltas-tol cap 0 {np.maximum(0, deltas - tol)}")
        '''
        
        delta_list.append(np.maximum(0, deltas - tol))
    PCE = np.mean(correct)
    PCE_class = np.mean(correct, axis=0)
    delta_class = np.mean(delta_list, axis=0)
    return PCE, PCE_class, delta_class

if __name__ == '__main__':


    split = 1
    n_cpu = 6

    args = get_args()
    device = check_device()

    print(f"args = {args}")
    model = get_model(args.model)

    if not args.val:
        data_loader = get_test_data_loader(model)
    else:
        data_loader = get_val_data_loader(model)

    if args.weight==None:
        print("get default weight--")
        weight_file = 'models/{}.pth.tar'.format(get_model_name(model))
    else:
        print("get customized weights")
        weight_file = args.weight

    if weight_file is not None:
        save_dict = torch.load(weight_file)
    else:
        raise ValueError("Weight file path is None!")
    save_dict = torch.load(weight_file)
    print(f"Evaluating model {args.model} on weight {weight_file}")
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        PCE, PCE_class, delta_class = eval(model, data_loader, split, args.seqlen, n_cpu, True)
    PCE = PCE.round(decimals=3).tolist()
    PCE_class = PCE_class.round(decimals=3).tolist()
    delta_class = delta_class.round(decimals=3).tolist()
    print('Average PCE: {} '.format(PCE))
    print('class PCEs {} '.format(PCE_class))
    print('class deltas {} '.format(delta_class))


