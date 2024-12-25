import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from augmented_datasets import AugmentedDSECDataset
from multiscaleloss import flow_error_dsec_supervised
from util import StatTracker
import os
import sys

import cv2
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
from copy import deepcopy

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def train(train_loader, model, optim, epoch, log_file, no_grad_split, grad_scalar, n_split):
    loss_stat = StatTracker() 

    if model.module.__class__.__name__ in ['EfficientSpikeEVFlowNet']:
        # EfficientSpikeEVFlowNet returns predictions + prev_v1s + prev_z1s
        #  prev_v1s & prev_z1s have size (num_encoders * 2 + num_resblock * 2)
        state_len = model.module.num_encoders * 2 + model.module.num_res_blocks * 2
    elif model.module.__class__.__name__ in ['LSTMEVFlowNet']:
        # LSTMEVFlowNet returns predictions + prev_c1s + prev_h1s
        #  prev_c1s & prev_h1s have size (num_encoders * 2)
        state_len = model.module.num_encoders * 2

    pbar = tqdm(train_loader)
    for batch_idx, data in enumerate(pbar, 0):
        # event_repr has a dimension (batch_size, num_bins, num_polarities, height, width)
        # flow has a dimension (batch_size, direction, height, width)
        # flow_mask has a dimension (batch_size, height, width)
        event_reprs, gt_flows, gt_flow_masks = data
        gt_flows = gt_flows.cuda()

        # Switch model to testing mode
        model.eval()
        if no_grad_split == 0:
            last_netstate_1, last_netstate_2 = None, None
        else:
            with torch.no_grad():
                outps = model(event_reprs[:, :no_grad_split])
            if model.module.__class__.__name__ in ['EfficientSpikeEVFlowNet', 'LSTMEVFlowNet']:
                outp_len = no_grad_split
                last_netstate_1, last_netstate_2 = outps[outp_len:outp_len+state_len], outps[outp_len+state_len:outp_len+(2*state_len)]
                assert len(outps) == outp_len+(2*state_len)
            else:
                assert False, 'Network with memory doesnt have a proper handle or no_grad_split is specified for feed-forward network'

        # Switch model to training mode
        model.train()
        # Process output based on the network type
        if model.module.__class__.__name__ in ['EfficientSpikeEVFlowNet', 'LSTMEVFlowNet']:
            outp_len = n_split
            outps = model(event_reprs[:, no_grad_split:], last_netstate_1, last_netstate_2)
            assert len(outps) == outp_len+(2*state_len)
        elif model.module.__class__.__name__ in ['NonSpikingEVFlowNet', 'SpikeFlowNet', 'AdaptiveFlowNet']:
            outp_len = 4
            outps = model(event_reprs)
            assert len(outps) == outp_len
        elif model.module.__class__.__name__ in ['AdaptiveFlowNet2S']:
            outp_len = 2
            outps = model(event_reprs)
            assert len(outps) == outp_len
        pred_flows = outps[outp_len - 1]

        gt_flow_masks = gt_flow_masks.unsqueeze(dim=1).expand(gt_flows.shape).cuda()
        all_pixel_errors = (gt_flows - pred_flows)**2
        valid_pixel_errors = all_pixel_errors[gt_flow_masks]
        avg_loss = torch.mean(valid_pixel_errors)

        # Compute error
        error = gt_flows - pred_flows  # Shape: (B, 2, H, W)

        # Compute neighboring correlation penalty
        # Extract the two error components (u and v directions)
        error_u = error[:, 0:1, :, :]  # Shape: (B, 1, H, W)
        error_v = error[:, 1:2, :, :]  # Shape: (B, 1, H, W)

        # Compute shifts for both components
        shifts = {
            'up': lambda x: torch.nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='constant', value=0),
            'down': lambda x: torch.nn.functional.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode='constant', value=0),
            'left': lambda x: torch.nn.functional.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='constant', value=0),
            'right': lambda x: torch.nn.functional.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='constant', value=0),
        }

        # Compute dot products of (u, v) with their shifted neighbors
        correlation_penalty = 0
        for shift_name, shift_fn in shifts.items():
            shifted_u = shift_fn(error_u)
            shifted_v = shift_fn(error_v)

            # Compute dot product of (u, v) with their shifted counterparts
            corr_u = (error_u * shifted_u).mean()
            corr_v = (error_v * shifted_v).mean()
            correlation_penalty += corr_u + corr_v

        # Normalize penalty by the number of shifts
        correlation_penalty /= len(shifts)

        # Combine losses
        lambda_correlation = 0.1  # Weight for the correlation penalty
        print(avg_loss)
        print(lambda_correlation * correlation_penalty)
        total_loss = avg_loss + lambda_correlation * correlation_penalty

        # compute gradient and do optimization step
        optim.zero_grad()
        # avg_loss.backward()
        # optim.step()
        grad_scalar.scale(total_loss).backward()
        grad_scalar.step(optim)
        grad_scalar.update()

        loss_stat.update(valid_pixel_errors.detach())

        if batch_idx % 20 == 0:
            pbar.set_description("Training - Epoch: {} Loss: {}".format(epoch+1, loss_stat))
    log_file.write('Epoch: {} Loss: {}\n'.format(epoch, loss_stat))

    return str(loss_stat)


def validate(test_loader, model, mode, visualize, save_dir, n_split, no_grad_ts):

    # Switch model to the testing mode
    model.eval()

    error_stat = StatTracker()
    masked_error_stat = StatTracker() 
    gt_flow_stat = StatTracker()

    last_netstate_1 = None
    last_netstate_2 = None
    total_errors = 0
    total_pe1 = 0
    total_pe2 = 0
    total_pe3 = 0
    total_pe4 = 0
    total_pe5 = 0

    pred_list = []
    gt_list = []
    mask_list = []

    if model.module.__class__.__name__ in ['EfficientSpikeEVFlowNet']:
        outp_len = n_split * (1 + no_grad_ts) if mode == 'test_w_reset' else n_split
        # EfficientSpikeEVFlowNet returns predictions + prev_v1s + prev_z1s
        #  prev_v1s & prev_z1s have size (num_encoders * 2 + num_resblock * 2)
        state_len = model.module.num_encoders * 2 + model.module.num_res_blocks * 2
    elif model.module.__class__.__name__ in ['LSTMEVFlowNet']:
        outp_len = n_split * (1 + no_grad_ts) if mode == 'test_w_reset' else n_split
        # LSTMEVFlowNet returns predictions + prev_c1s + prev_h1s
        #  prev_c1s & prev_h1s have size (num_encoders * 2)
        state_len = model.module.num_encoders * 2
    elif model.module.__class__.__name__ in ['NonSpikingEVFlowNet', 'SpikeFlowNet', 'AdaptiveFlowNet']:
        outp_len = 4
    elif model.module.__class__.__name__ in ['AdaptiveFlowNet2S']:
        outp_len = 2

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, data in enumerate(pbar, 0):
            # Conditional check for processing data
            if mode == 'test_wo_reset':
                event_reprs, gt_flows, gt_flow_masks, reset, valid = data
                # Reset last_netstate_2 and last_netstate_1 when model reset
                if reset:
                    last_netstate_2 = None
                    last_netstate_1 = None
            else:
                # event_repr has a dimension (batch_size, num_bins, num_polarities, height, width)
                # flow has a dimension (batch_size, direction, height, width)
                event_reprs, gt_flows, gt_flow_masks = data

            # Conditional check for running model
            if model.module.__class__.__name__ in ['EfficientSpikeEVFlowNet', 'LSTMEVFlowNet']:
                outps = model(event_reprs, last_netstate_1, last_netstate_2)
                if mode == 'test_wo_reset':
                    last_netstate_1, last_netstate_2 = outps[outp_len:outp_len+state_len], \
                                                       outps[outp_len+state_len:outp_len+(2*state_len)]
                assert len(outps) == outp_len+(2*state_len)
            elif model.module.__class__.__name__ in ['NonSpikingEVFlowNet', 'SpikeFlowNet', 'AdaptiveFlowNet', 'AdaptiveFlowNet2S']:
                outps = model(event_reprs)

            # Add evaluation results to the stat tracking
            if mode == 'test_wo_reset' and valid:
                pred_flows = outps[:outp_len]
                if model.module.__class__.__name__ in ['AdaptiveFlowNet2S']:
                    pred_list.append(pred_flows[1])
                    gt_list.append(gt_flows[1])
                    mask_list.append(gt_flow_masks[1])
                elif model.module.__class__.__name__ in ['NonSpikingEVFlowNet']:
                    pred_list.append(pred_flows[2])
                    gt_list.append(gt_flows[2])
                    mask_list.append(gt_flow_masks[2])
                else:
                    pred_list.append(pred_flows[3])
                    gt_list.append(gt_flows[3])
                    mask_list.append(gt_flow_masks[3])
                # for pred_flow, gt_flow, gt_flow_mask in zip(pred_flows, gt_flows, gt_flow_masks):
                #     valid_pixel_errors, n_errors, \
                #     n_pe1, n_pe2, n_pe3, n_pe4, n_pe5 = \
                #         flow_error_dsec_supervised(gt_flow, gt_flow_mask, pred_flow, event_reprs,
                #                                    error_stat, masked_error_stat, gt_flow_stat)
                #     total_errors += n_errors
                #     total_pe1 += n_pe1
                #     total_pe2 += n_pe2
                #     total_pe3 += n_pe3
                #     total_pe4 += n_pe4
                #     total_pe5 += n_pe5

            elif mode != 'test_wo_reset':
                pred_flows = outps[outp_len - 1]
                if model.module.__class__.__name__ in ['NonSpikingEVFlowNet']:
                    pred_list.append(pred_flows[0])
                    gt_list.append(gt_flows[0])
                    mask_list.append(gt_flow_masks[0])
                else:
                    pred_list.append(pred_flows[3])
                    gt_list.append(gt_flows[3])
                    mask_list.append(gt_flow_masks[3])
                valid_pixel_errors, n_errors, \
                n_pe1, n_pe2, n_pe3, n_pe4, n_pe5 = \
                    flow_error_dsec_supervised(gt_flows, gt_flow_masks, pred_flows, event_reprs,
                                               error_stat, masked_error_stat, gt_flow_stat)
                total_errors += n_errors
                total_pe1 += n_pe1
                total_pe2 += n_pe2
                total_pe3 += n_pe3
                total_pe4 += n_pe4
                total_pe5 += n_pe5

                if visualize:
                    np_gt_flow = gt_flows[0].numpy()
                    np_gt_flow_mask = gt_flow_masks[0].float().numpy()
                    np_nonzero_event_mask = (torch.sum(event_reprs[0], dim=[0, 1]) > 0).numpy()
                    masked_gt_flow_rgb = flow_viz_np(np_gt_flow[0, :, :] * np_gt_flow_mask * np_nonzero_event_mask,
                                                     np_gt_flow[1, :, :] * np_gt_flow_mask * np_nonzero_event_mask)
                    cv2.imshow('Visualize ground truth flow', cv2.cvtColor(masked_gt_flow_rgb, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)  # wait time in millisecond unit

    pred = torch.cat(pred_list, dim = 0)
    gt = torch.cat(gt_list, dim = 0)
    mask = torch.cat(mask_list, dim = 0)

    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()

    outdir = save_dir

    np.save(os.path.join(outdir, 'prediction.npy'), pred_np)
    np.save(os.path.join(outdir, 'gt.npy'), gt_np)
    np.save(os.path.join(outdir, 'mask.npy'), mask_np)
    # print(error_stat)
    # print(masked_error_stat)
    # print(gt_flow_stat)
    # print('---- NP ----- ')
    # print(f'{total_errors:.3e}, '
    #       f'{total_pe1 / total_errors:.3e}, '
    #       f'{total_pe2 / total_errors:.3e}, '
    #       f'{total_pe3 / total_errors:.3e}, '
    #       f'{total_pe4 / total_errors:.3e}, '
    #       f'{total_pe5 / total_errors:.3e}')
    # print('--- nonzero and total output count ---')
    # print(', '.join(['{:.3e}'.format(float(each.avg)) for each in model.module.n_nonzero_inp_trackers]))
    # print(', '.join(['{:.3e}'.format(float(each.avg)) for each in model.module.n_inp_trackers]))


if __name__ == '__main__':
    # Parser for setting hyperparameters during training and testing 
    parser = argparse.ArgumentParser(description='Optical flow prediction training script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Name of the result directory')
    parser.add_argument('--arch', default='EVFlowNet', help='Model architecture')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--n-epochs', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--dataset-dir', default=None, help='Path to pre-processed dataset')

    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'], help='Solver algorithms')
    parser.add_argument('--save-interval', default=1, type=int, help='Save interval')
    parser.add_argument('--mode', default='test_w_reset', type=str, help='Evaluate model on validation set')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite exisiting file?')
    parser.add_argument('--model-path', default=None, help='Path to saved model')

    parser.add_argument('--use-scheduler', action='store_true', help='Use learning rate scheduler?')
    parser.add_argument('--milestones', default=[5, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170],
                        nargs='*', help='epochs at which learning rate is divided by 2')

    parser.add_argument('--dt', type=int, default=1, help='Frame difference for computing a photometric loss')
    parser.add_argument('--n-split', type=int, default=10, help='Number of bins for events representation')
    parser.add_argument('--model-options', default='', type=str,
                        help='Number of epoch to be trained with an incremental sequence length')

    parser.add_argument('--no-grad-ts', default=0, type=int,
                        help='Number of dt to run the model without backpropagtion during training (will be multiplied with n_split)')

    parser.add_argument('--visualize', action='store_true', help='Visualize grouth truth flow and prediction')

    args = parser.parse_args()
    assert args.mode in ['train', 'test_w_reset', 'test_wo_reset'], '{} is invalid mode'.format(args.mode)

    if args.visualize:
        assert args.bs == 1, 'Batch size must be 1 for visualization'

    # Dataset and groud truth paths 
    dataset_dir = os.path.abspath(args.dataset_dir)

    save_dir = os.path.join(args.save_dir, 'dt{},tsplit{},{},{},e{},bs{},lr{:.0e}{}{}'.format(
        args.dt, args.n_split, args.arch, args.solver,
        args.n_epochs, args.bs,
        args.lr,
        '' if args.model_options == '' else ','+args.model_options.replace('"', '').replace('\'', '').replace(':', '-'),
        '' if args.no_grad_ts == 0 else ',ng{}'.format(args.no_grad_ts)))

    if args.mode == 'train':
        # There is no need to convert sample in tensor format to PIL image for transformation
        # Be careful with ToPILImage() if need to be used since it normalizes inputs by default
        #   Tensor with value above 1 is cliped to 1
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # Default random flip function cannot be used for DSEC dataset
            # since optical flows have both quantity and direction unlike image intensities
            # We create an option to randomly flip the sample as a part of dataset
            transforms.RandomCrop((128, 128))
        ])
        train_dataset = AugmentedDSECDataset(dataset_dir, dt=args.dt, n_split=args.n_split,
                                                        transform=train_transform, random_flip=True, mode='train',
                                                        n_prefix_event_repr=args.no_grad_ts)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                  pin_memory=False)
    elif 'test' in args.mode:
        test_transform = transforms.Compose([
            transforms.CenterCrop((128, 128))
        ])
        test_dataset = AugmentedDSECDataset(dataset_dir, dt=args.dt, n_split=args.n_split,
                                                       transform=test_transform, random_flip=False, mode=args.mode,
                                                       n_prefix_event_repr=args.no_grad_ts)
        print('event_repr.shape:', test_dataset[0][0].shape)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1 if args.mode == 'test_wo_reset' else args.bs,
                                 shuffle=False, num_workers=args.workers,
                                 pin_memory=False)

    model_options = ast.literal_eval('{'+args.model_options+'}')
    if args.arch in ['NonSpikingEVFlowNet']:
        for key, default_val in {'num_pols': 2 * args.n_split, 'num_encoders': 4, 'num_res_blocks': 0,
                                 'norm': 'bn-all-ts', 'c_mul': 2, 'num_base_c': 32}.items():
            if key not in model_options:
                model_options[key] = default_val
    elif args.arch in ['SpikeFlowNet']:
        for key, default_val in {'num_pols': 2, 'batch_norm': False, 'threshold': 0.75, 'n_split': args.n_split}.items():
            if key not in model_options:
                model_options[key] = default_val
    elif args.arch in ['AdaptiveFlowNet']:
        for key, default_val in {"num_pols": 2, "base_channels": 64, "batchNorm": False, "learn_thresh": True,
                                 "learn_leak": True, "ithresh": 1.0, "ileak": 1.0, "reset_mechanism": "soft",
                                 "per_channel": 0}.items():
            if key not in model_options:
                model_options[key] = default_val
    elif args.arch in ['AdaptiveFlowNet2S']:
        for key, default_val in {"num_pols": 2, "base_channels": 64*4, "batchNorm": False, "learn_thresh": True,
                                 "learn_leak": True, "ithresh": 1.0, "ileak": 1.0, "reset_mechanism": "soft",
                                 "per_channel": 0}.items():
            if key not in model_options:
                model_options[key] = default_val
    elif args.arch in ['EfficientSpikeEVFlowNet', 'LSTMEVFlowNet']:
        for key, default_val in {'num_pols': 2, 'num_encoders': 4, 'num_res_blocks': 0, 'norm': 'bn-all-ts', 'c_mul': 2,
                                 'num_base_c': 32}.items():
            if key not in model_options:
                model_options[key] = default_val
    else:
        print(args.arch)
        raise NotImplementedError

    model = models.__dict__[args.arch](model_options)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)['model'])
        print("=> Restore the pre-trained model '{}' from {}".format(args.arch, args.model_path))

    model = torch.nn.DataParallel(model).cuda()

    if 'test' in args.mode:
        validate(test_loader=test_loader, model=model, mode=args.mode, visualize=args.visualize,
                 save_dir=args.save_dir, n_split=args.n_split, no_grad_ts=args.no_grad_ts)

    elif args.mode == 'train':
        if os.path.isfile(save_dir+'.log') and not args.overwrite:
            print('File {} exists!'.format(save_dir+'.log'))
            sys.exit(0)
        log_file = open(save_dir+'.log', 'w', buffering=1)

        # Record input argument
        print('=> Input command ' + ' '.join(sys.argv))
        log_file.write(' '.join(sys.argv) + '\n')

        # Make a directory to save files if it doesn't exist 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('=> Results will be saved to {}'.format(save_dir))

        assert (args.solver in ['adam', 'sgd'])
        print('=> Setting {} solver'.format(args.solver))
        optim = None
        if args.solver == 'adam':
            optim = torch.optim.Adam(model.parameters(), args.lr)
        elif args.solver == 'sgd':
            optim = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)

        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args.milestones, gamma=0.1)
            print("=> Using learning rate scheduler")

        grad_scalar = torch.cuda.amp.GradScaler()

        for epoch in range(args.n_epochs):
            train_loss = train(train_loader=train_loader, model=model, optim=optim, epoch=epoch,
                               log_file=log_file, no_grad_split=args.n_split*args.no_grad_ts, grad_scalar=grad_scalar, 
                               n_split=args.n_split)

            if args.use_scheduler:
                scheduler.step()

            # Save the model for testing every N epochs
            if (epoch + 1) % args.save_interval == 0:
                torch.save({'epoch': epoch+1, 'arch': args.arch, 'optim': deepcopy(optim.state_dict()),
                            'model': deepcopy(model.module.state_dict())},
                           os.path.join(save_dir, 'checkpoint_ep{}.pt'.format(epoch+1)))

            torch.cuda.empty_cache()
            
        log_file.close()
