# Test
import torch
import time
from network.ACALF import ACALF
from common.config import parse_opts
from common.logger import Logger, AverageMeter
from data.dataset import FSSDataset
from common import utils
from common.evaluation import Evaluator
import torch.optim as optim
from torch import nn
import os
import datetime
import cv2

os.environ['CUDA_VISIBLE_DEVICES']= '0, 1, 2, 3, 4, 5'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
local_rank = int(os.environ["LOCAL_RANK"])
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main_process(local_rank):
    return not args.distributed or (args.distributed and (local_rank == 0))


def train(epoch, model, dataloader, optimizer, cross_entropy_loss, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    if args.distributed:
        if training:
            model.module.train()
            model.module.feat_extractor.eval()
        else:
            model.module.eval()
    else:
        model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        
        # 1. forward pass
        batch = utils.to_cuda(batch)

        if training:
            logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
        else:
            logit_mask = model.module(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
        pred_mask = logit_mask.argmax(dim=1)
        
       # 2. Compute loss & update model parameters
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = batch['query_mask'].view(bsz, -1).long()
        loss = cross_entropy_loss(logit_mask, gt_mask)
        

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        if main_process(local_rank):
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=20)
    
    # Write evaluation results
    if main_process(local_rank):
        average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

def main():
    global args
    args = parse_opts()

    # model initialization
    if args.backbone == 'swin-b':
        _feat_channels = [128, 256, 512, 1024]
    if args.backbone == 'swin-l':
        _feat_channels = [192, 384, 768, 1536]
    if args.backbone == 'resnet101' or args.backbone == 'resnet50':
        _feat_channels = [256, 512, 1024, 2048]
    if args.backbone == 'vgg':
        _feat_channels = [256, 512, 512, 512]

    model = ACALF(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False,
                  feat_channels=_feat_channels, hidden_dim=args.hidden_dim, num_queries=args.num_queries,
                  nhead=args.nhead, dec_layers=args.dec_layers, conv_dim=args.conv_dim, mask_dim=args.mask_dim)

    optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.05}])
    cross_entropy_loss = nn.CrossEntropyLoss()
    Evaluator.initialize()
    FSSDataset.initialize(img_size=384, use_original_imgsize=False)

    if torch.cuda.device_count() > 1:
        args.distributed = True
    else:
        args.distributed = False

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1000000))
        # local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        # global device
        device = torch.device("cuda", local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if args.resume != '':
            params = model.state_dict()
            state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
            for k1, k2 in zip(list(state_dict.keys()), params.keys()):
                state_dict[k2] = state_dict.pop(k1)
            model.load_state_dict(state_dict)
        if local_rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad is True:
                    print(name)
                else:
                    print("no grad", name)
            Logger.initialize(args, training = True)
            Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    else:
        device = torch.device("cuda")
        model.to(device)
        model.train()
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    dataloader_trn = FSSDataset.build_dataloader(args.datapath, args.test_num, args.distributed, args.benchmark, args.bsz, args.nworker, args.fold,
                                                 'trn', training=True)
    dataloader_val = FSSDataset.build_dataloader(args.datapath, args.test_num, args.distributed, args.benchmark, args.bsz, args.nworker, args.fold,
                                                 'val', training=True)
    
    for epoch in range(args.nepoch):
        if args.distributed:
            dataloader_trn.sampler.set_epoch(epoch)
        _, _, _ = train(epoch, model, dataloader_trn, optimizer, cross_entropy_loss, training=True)

        # save model every 10 epochs
        if main_process(local_rank) and epoch % 10 == 0:
            with torch.no_grad():
                _, current_val, _ = train(epoch, model, dataloader_val, optimizer, cross_entropy_loss, training=False)
            Logger.save_model_miou(model, epoch, current_val)
                 

if __name__ == "__main__":
    main()
