import torch.nn as nn
import torch
from network.ACALF import ACALF
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10000"


def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)

    average_meter = AverageMeter(dataloader.dataset)
    epoch = args.test_epoch
    start_time = time.time()
    for i in range(args.test_epoch):
        start_time = time.time()
        for idx, batch in enumerate(dataloader):

            batch = utils.to_cuda(batch)
            if args.nshot == 1:
                logit_mask = model.module(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
                pred_mask = logit_mask.argmax(dim=1)    
            else:
                if args.pre_average: # average features directly after feature extraction
                    logit_mask = model.module(batch['query_img'],batch['support_imgs'], batch['support_masks'], args.nshot)
                    pred_mask = logit_mask.argmax(dim=1)
                elif args.post_average: # treat each shot as 1-shot then average after obtaining masks
                    logit_masks = []
                    for k in range(args.nshot):
                        temp_mask = model.module(batch['query_img'], batch['support_imgs'][:, k], batch['support_masks'][:, k], 1)
                        logit_masks.append(temp_mask)
                    logit_mask = torch.stack(logit_masks, dim=1).mean(dim=1)
                    pred_mask = logit_mask.argmax(dim=1)
                elif args.vote:
                    logit_mask_agg = 0
                    for s_idx in range(args.nshot):
                        logit_mask = model.module(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], 1)
                        logit_mask_agg += logit_mask.argmax(dim=1).clone()
                    bsz = logit_mask_agg.size(0)
                    max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
                    max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
                    max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
                    pred_mask = logit_mask_agg.float() / max_vote
                    pred_mask[pred_mask < 0.5] = 0
                    pred_mask[pred_mask >= 0.5] = 1

            assert pred_mask.size() == batch['query_mask'].size()
            bsz = logit_mask.size(0)
            logit_mask = logit_mask.view(bsz, 2, -1)
            gt_mask = batch['query_mask'].view(bsz, -1).long()

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)

            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), epoch=i, write_batch_idx=1)

            # Logger.info(area_inter[1].float() / area_union[1].float())
            torch.cuda.empty_cache()

    # Write evaluation results
    average_meter.write_result('Test', epoch)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':
    args = parse_opts()
    Logger.initialize(args, training=False)

    # Model initialization
    if args.backbone == 'swin-b':
        _feat_channels = [128, 256, 512, 1024]
    if args.backbone == 'swin-l':
        _feat_channels = [192, 384, 768, 1536]
    if args.backbone == 'resnet101' or args.backbone == 'resnet50':
        _feat_channels = [256, 512, 1024, 2048]

    print("Initializing Model...")
    model = ACALF(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False, feat_channels=_feat_channels, hidden_dim=args.hidden_dim,
            num_queries=args.num_queries, nhead=args.nhead, dec_layers=args.dec_layers, conv_dim=args.conv_dim, mask_dim=args.mask_dim)
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    print("Loading Model...")
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)
    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
    model.load_state_dict(state_dict)
    model.module.eval()

    # Helper classes (for testing) initialization
    Evaluator.initialize()

    # Dataset initialization
    print("Preparing Dataset...")
    FSSDataset.initialize(img_size=384, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, args.test_dataset, args.bsz, args.nworker, args.fold, 'val', args.nshot, training=False)
    
    # Test
    print("Evaluating...")
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('%s mIoU: %5.2f \t FB-IoU: %5.2f' % (args.test_dataset, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')\
    
    print("Writing Results...")
    with open('results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = '%s-%s %sshot_test' % (args.backbone,  args.test_dataset, args.nshot)
        acc_str = '%d Test Miou = %4.2f ' % (args.test_num, test_miou)
        f.write('Time: %s, Setting: %s, Miou: %s \n' % (timestamp, exp_setting, acc_str))
    print("Done!")

    
    
