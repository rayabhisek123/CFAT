###################---------Import---------###################
import os 
import argparse
import torch
import time
import datetime
import random
import cv2
import glob
import requests
import utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
import skimage.color as sc

from collections import OrderedDict
from importlib import import_module
from tqdm import tqdm
from torchsummary import summary
from ptflops import get_model_complexity_info
from data import DIV2K_train, DIV2K_valid, Set5_val
from torch.utils.data import DataLoader
import torch.optim as optim
from util_calculate_psnr_ssim import bgr2ycbcr, calculate_psnr, calculate_ssim, calculate_psnrb

#from model.swin2sr import Swin2SR as net   #patch_size=192
#from model import hat
from model import cfat
# from Model import MFSR

torch.backends.cudnn.benchmark = True

###################---------Arguments---------###################
# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=700, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=[350, 550, 600, 650], help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5, help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loading")
parser.add_argument("--root", type=str, default="./Datasets/DIV2K/", help='dataset directory')
parser.add_argument("--n_train", type=int, default=800, help="number of training set")
parser.add_argument("--n_val", type=int, default=5, help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4, help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=256, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3, help="number of color channels to use")
parser.add_argument("--in_channels", type=int, default=32, help="number of channels for transformer")
parser.add_argument("--n_layers", type=int, default=3, help="number of FETB uits to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')
parser.add_argument("--channel_in_DE", type=int, default=3)
parser.add_argument("--channel_base_DE", type=int, default=8)
parser.add_argument("--channel_int", type=int, default=16)
parser.add_argument("--output_dir", type=str, default="Put the adress of checkpoint directory here")
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--log_period", type=int, default=10)
parser.add_argument("--checkpoint_period", type=int, default=10)
#parser.add_argument("--eval_period", type=int, default=2)
parser.add_argument('--folder_lq', type=str, default="./TestData_LR/", help='input low-quality test image folder')
parser.add_argument('--folder_gt', type=str, default="./TestData_GT/", help='input ground-truth test image folder')
parser.add_argument("--output_folder", type=str, default="./TestData_OUT/")
parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr')
parser.add_argument('--save_img_only', default=False, action='store_true', help='save image and do not evaluate')
parser.add_argument('--tile', type=int, default=64, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=16, help='Overlapping of different tiles')
args = parser.parse_args()
print(args)


###################---------Random_Seed---------###################
if args.seed:
    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
else:
    seed_val = random.randint(1, 10000)
    print("Ramdom Seed: ", seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = False



###################---------Environment---------###################
cuda = args.cuda
device = torch.device('cuda:0' if cuda else 'cpu')
gpus=[0]    #[0, 1, 2, 4] for batch size of 12
def ngpu(gpus):
    """count how many gpus used"""
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


###################---------Model---------###################
print("===> Building models")
torch.cuda.empty_cache()
args.is_train = True

##Model::ESRT
#model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
#                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
#model = hat.HAT()
model = cfat.CFAT()

###################---------Loss_Function---------###################
l1_criterion = nn.L1Loss()

###################---------Optimizer---------###################
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.step_size, gamma=args.gamma)

###################---------.to(Device)---------###################
print("===> Setting GPU")
if cuda:
    #print(device, gpus)
    #input()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    l1_criterion = l1_criterion.to(device)

###################---------Load_model_to_resume_training---------###################
begin_epoch=1
train_loss={}
checkpoint_file=os.path.join(args.output_dir, 'CVPR_250.pth')
#checkpoint=torch.load(checkpoint_file)
#model.load_state_dict(checkpoint['state_dict'])
if os.path.exists(checkpoint_file):
    #   print('')
      checkpoint=torch.load(checkpoint_file)
      begin_epoch = checkpoint['epoch']
      train_loss=checkpoint['train_loss']
      print('loading state dict')
      model.load_state_dict(checkpoint['state_dict'])
      print('loaded state dict')
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
model.eval()
save_dir = f'img_sr/swin2sr_{args.task}_x{args.scale}'
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
if args.save_img_only:
    folder = args.folder_lq
else:
    folder = args.folder_gt
border = args.scale
window_size = 16

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        if args.save_img_only:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
        else:
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.            
        
    elif args.task in ['compressed_sr']:
        if args.save_img_only:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
        else:
            img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.        

    # 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr', 'lightweight_sr_infer']:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 006 grayscale JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt.ndim != 2:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_jpeg_car']:
        img_gt = cv2.imread(path)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 1)
        img_gt = img_gt.astype(np.float32)/ 255.
        img_lq = img_lq.astype(np.float32)/ 255.

    return imgname, img_lq, img_gt



def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                #print(in_patch.shape)
                #input()
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


os.makedirs(save_dir, exist_ok=True)
test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []
test_results['psnrb'] = []
test_results['psnrb_y'] = []
psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
    # read image
    imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
    #print(imgname, img_lq.shape, img_gt.shape)
    #input()
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = test(img_lq, model, args, window_size)
        
        if args.task == 'compressed_sr':
            output = output[0][..., :h_old * args.scale, :w_old * args.scale]
        else:
            output = output[..., :h_old * args.scale, :w_old * args.scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(f'{save_dir}/{imgname}_Swin2SR.png', output)

        
    # evaluate psnr/ssim/psnr_b
    if img_gt is not None:
        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
        img_gt = np.squeeze(img_gt)

        psnr = calculate_psnr(output, img_gt, crop_border=border)
        ssim = calculate_ssim(output, img_gt, crop_border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        if img_gt.ndim == 3:  # RGB image
            psnr_y = calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
            ssim_y = calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
        if args.task in ['jpeg_car', 'color_jpeg_car']:
            psnrb = calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=False)
            test_results['psnrb'].append(psnrb)
            if args.task in ['color_jpeg_car']:
                psnrb_y = calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnrb_y'].append(psnrb_y)
        print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNRB: {:.2f} dB;'
              'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; PSNRB_Y: {:.2f} dB.'.
              format(idx, imgname, psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y))
    else:
        print('Testing {:d} {:20s}'.format(idx, imgname))

# summarize psnr/ssim
if img_gt is not None:
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
    if img_gt.ndim == 3:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
    if args.task in ['jpeg_car', 'color_jpeg_car']:
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
        print('-- Average PSNRB: {:.2f} dB'.format(ave_psnrb))
        if args.task in ['color_jpeg_car']:
            ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
            print('-- Average PSNRB_Y: {:.2f} dB'.format(ave_psnrb_y))

