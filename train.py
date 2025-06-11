
from SwingNet import EventDetector
from golfdb.util import *
import torch
from torchvision import transforms
import os
from trainer import train
from utils import get_args, check_device
from detector_factory import get_model
from dataloader_factory import get_training_data_loader
from plot import plot

if __name__ == '__main__':
    args = get_args()
    device = check_device()

    print(f"args = {args}")
    model = get_model(args.model, args.drop)

    if args.start_weight != None:
        save_dict = torch.load(args.start_weight)
        model.load_state_dict(save_dict['model_state_dict'])    
        print(f"Loaded starting weight at {args.start_weight}")
    model.to(device)

    data_remote = True if args.data_remote != None else False
    training_data_loader = get_training_data_loader(model=model,
                                                    batch_size=args.batch, 
                                                    seq_length=args.seqlen, 
                                                    data_remote=data_remote)
    



    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    print(model)
    for name, param in model.named_parameters():
        print(f"{name} requires_grade={param.requires_grad} param count {param.numel()}")

    print(f"Model Parameters: [{total_trainable_params} trainable / {total_params} total]")

    loss, acc, event_count = train(model, 
                      training_data_loader,
                      iterations=args.iterations, 
                      batch_size=args.batch, 
                      seq_length=args.seqlen,
                      lr=args.lr)
    print(f"event_count {event_count}")
    plot(loss, acc)
