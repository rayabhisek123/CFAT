###################---------Import---------###################
import os 
import argparse
import torch
import time
import datetime
import random
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

from model import cfat
#from model_wv_abhi_gau import hat
# from Model_MFSR import MFSR
#from model_wacv import esrt
from data import DIV2K_train, DIV2K_valid, Set5_val
from torch.utils.data import DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True


###################---------Arguments---------###################
# Training settings
parser = argparse.ArgumentParser(description="CFAT")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=500, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=[225, 350, 400, 450], help="learning rate decay per N epochs")
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
parser.add_argument("--in_channels", type=int, default=72, help="number of channels for transformer")
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
parser.add_argument("--output_dir", type=str, default="./pretrained_models/")
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--log_period", type=int, default=10)
parser.add_argument("--checkpoint_period", type=int, default=10)
parser.add_argument("--eval_period", type=int, default=2)
parser.add_argument("--validtext", type=str, default="./pretrained_models/Validation.txt", help='text file to store validation results')
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
    torch.backends.cudnn.benchmark = True
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


###################---------Weights_Initiazation---------###################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('PixelShuffle') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if m.bias is not None:
        m.bias.data.fill_(0)



###################---------Environment---------###################
cuda = args.cuda
device = torch.device('cuda:0' if cuda else 'cpu')
gpus=[0]    #[0, 1, 2, 4] for batch size of 12
def ngpu(gpus):
    """count how many gpus used"""
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)



###################---------Dataset---------###################
print("===> Loading datasets")
trainset = DIV2K_train.div2k(args)  #Initialize
#testset = DIV2K_valid.div2k(args)
testset = Set5_val.DatasetFromFolderVal("./Set5_gt/", "./Set5/", args.scale)
#testset = Set5_val.DatasetFromFolderVal("./Datasets/DIV2K/DIV2K_valid/DIV2K_valid_HR/",
#                                       "./Datasets/DIV2K/DIV2K_valid/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X{}/".format(args.scale), args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads * ngpu(gpus), batch_size=args.batch_size, shuffle=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads * ngpu(gpus), batch_size=args.testBatchSize, shuffle=False, drop_last=True)



###################---------Model---------###################
print("===> Building models")
torch.cuda.empty_cache()
args.is_train = True

##Model::ESRT
#args.is_train = True
model = cfat.CFAT()

##Model::MFSR
#arg = {'resolutions': [2, 3, 4], 'scale_256': [4, 2, 1], 'scale_64': [1, 2, 4], 'resolution_in': 4, 'num_unit_BC': 1, 'num_unit_AC': 2, 'channel_in_DE': 3, 'channel_base_DE':8}
# model = MFSR.Model(args)


###################---------Loss_Function---------###################
l1_criterion = nn.L1Loss()


###################---------Optimizer---------###################
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.base_lr, eta_min=0.01 * args.base_lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
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
checkpoint_file=os.path.join(args.output_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_file):
      checkpoint=torch.load(checkpoint_file)
      begin_epoch = checkpoint['epoch']
      train_loss=checkpoint['train_loss']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))



###################---------Define_Training_Epoch---------###################
def train(epoch):

    #utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    accum_iter=4          # Gradient accumulation

    ##Training_Iteration_Start
    start_time = time.time()
    model.train()
    loader = tqdm(training_data_loader)
    with tqdm(training_data_loader, unit="batch") as tepoch:
            
        for iteration, batch in enumerate(tepoch, 1):
            lr_tensor, hr_tensor = batch[0], batch[1]
            tepoch.set_description(f"Epoch {epoch}")
            if args.cuda:
                lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
                hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
            
            sr_tensor = model(lr_tensor)
            loss_l1 = l1_criterion(sr_tensor, hr_tensor)
            loss_sr = loss_l1
            optimizer.zero_grad()
            loss_sr.backward()
            optimizer.step()
            loss_meter.update(loss_sr.item(), lr_tensor.shape[0])
            tepoch.set_postfix(loss=loss_sr.item())
            torch.cuda.synchronize()
        
        
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        #For_Print
        if (iteration) % args.log_period == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(epoch, iteration, len(training_data_loader),loss_meter.avg))
    
    end_time = time.time()
    time_per_batch = (end_time - start_time) / (iteration)
    print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] Avg_Loss: {:.4f}".format(epoch, time_per_batch, training_data_loader.batch_size / time_per_batch, loss_meter.avg*accum_iter))
    ##Iteration_End
    
    txt_write = open(args.validtext, 'a')
    print("Epoch: {}  Learning rate: {:.5f}  Loss: {:.4f}".format(epoch, scheduler.optimizer.param_groups[0]['lr'], loss_meter.avg*accum_iter), file = txt_write)
    #txt_write.close()

    #Save_Loss
    train_loss[epoch]=loss_meter.avg*accum_iter
    
    #Save_Check-points
    states={
            'train_loss':train_loss,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
            }   
            
    torch.save(states, os.path.join(args.output_dir, 'checkpoint.pth')) 
    if epoch % args.checkpoint_period == 0 or epoch == args.nEpochs:
        torch.save(states, os.path.join(args.output_dir, 'checkpoint'+'_{}.pth'.format(epoch)))



###################---------valid_window---------###################

def valid_window(model, x, input_shape = 64): 
    scale = args.scale
    #x = torch.randn(1, 64, 130, 135)
    B, C, H, W = x.size() 
    x_pred = torch.zeros(B, C, scale*H, scale*W)
    H_Q, W_Q = H // input_shape, W// input_shape  
    row_list = [i*input_shape for i in range(0, H_Q+1)] 
    col_list = [i*input_shape for i in range(0, W_Q+1)]                   
    H_R, W_R = H % input_shape, W % input_shape      
    x_all = []
    for i in range(0, len(row_list)):
      for j in range(0, len(col_list)):
        if i<len(row_list)-1 and j<len(col_list)-1:
          x1 = x[:, :, row_list[i]:row_list[i+1], col_list[j]:col_list[j+1]]
          x_all.append(x1)
        if i==len(row_list)-1 and j<len(col_list)-1:
          x1 = x[:, :, H-input_shape:H, col_list[j]:col_list[j+1]]
          x_all.append(x1)
        elif i<len(row_list)-1 and j==len(col_list)-1:
          x1 = x[:, :, row_list[i]:row_list[i+1], W-input_shape:W]
          x_all.append(x1)
        elif i==len(row_list)-1 and j==len(col_list)-1:
          x1 = x[:, :, H-input_shape:H, W-input_shape:W]
          x_all.append(x1)
    # print("x_all",len(x_all))
    # input()
    lr_batch = torch.cat(x_all, dim=0)
    sr_batch = model(lr_batch) 
    sr_list = []
    sr_list.extend(sr_batch.chunk(len(sr_batch), dim=0))
    for i in range(0, len(row_list)):
      for j in range(0, len(col_list)):
        if i<len(row_list)-1 and j<len(col_list)-1:
          x_pred[:, :, scale*row_list[i]:scale*row_list[i+1], scale*col_list[j]:scale*col_list[j+1]] = sr_list[len(col_list)*i+j][:, :, :, :]
        if i==len(row_list)-1 and j<len(col_list)-1:
          x_pred[:, :, scale*(H-H_R):scale*H, scale*col_list[j]:scale*col_list[j+1]] = sr_list[len(col_list)*i+j][:, :, scale*(input_shape-H_R):scale*input_shape, :]
        elif i<len(row_list)-1 and j==len(col_list)-1:
          x_pred[:, :, scale*row_list[i]:scale*row_list[i+1], scale*(W-W_R):scale*W] = sr_list[len(col_list)*i+j][:, :, :, scale*(input_shape-W_R):scale*input_shape]
        elif i==len(row_list)-1 and j==len(col_list)-1:
          x_pred[:, :, scale*(H-H_R):scale*H, scale*(W-W_R):scale*W] = sr_list[len(col_list)*i+j][:, :, scale*(input_shape-H_R):scale*input_shape, scale*(input_shape-W_R):scale*input_shape]
    return x_pred

###################---------Define_Validation_Epoch---------###################
def valid(scale):
    model.eval()
    avg_psnr, avg_ssim = 0, 0
    test_loader = tqdm(testing_data_loader)
    for iteration, batch in enumerate(test_loader):   #testing_data_loader = 
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)         #[128, 3, 48, 48])
            hr_tensor = hr_tensor.to(device)         #[128, 3, 192, 192]
            #print('lr_tensor',lr_tensor.size())
            #input()

        lr_tensors = lr_tensor.chunk(len(lr_tensor), dim=0)
        lr_out_all = []
        with torch.no_grad():
            for lr_tensor in lr_tensors:
                lr_out = valid_window(model, lr_tensor)
                lr_out_all.append(lr_out)
            #sr_tensor = model(lr_tensor)
        
        lr_out_all = [lr_out_all[i].detach()[0] for i in range(len(lr_out_all))]
        lr_batch = torch.cat(lr_out_all, dim=0)
        sr_img = utils.tensor2np(lr_batch)
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        # print(im_pre.shape)
        # print(im_label.shape)
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    #print(len(testing_data_loader))
    #input()
    txt_write = open(args.validtext, 'a')
    print("Validation Results - Epoch: {}".format(epoch), file = txt_write)
    print("PSNR: {:.4f}".format(avg_psnr / len(testing_data_loader)), file = txt_write)
    print("SSIM: {:.4f}".format(avg_ssim / len(testing_data_loader)), file = txt_write)
    #txt_write.close()
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))



###################---------save_checkpoint---------###################
def save_checkpoint(epoch):
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))




###################---------Tot_Parameter_Count---------###################
def print_network(model):
    summary(model, (3, 64, 64))
    macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


print("===> Training")

##Computational_Complexity[Parameters_&_Flops]
print_network(model)

##Start_Total_Compuational_Time
code_start_time = datetime.datetime.now()
loss_meter = AverageMeter()   
timer = utils.Timer()

optimizer.zero_grad() #We are using gradient accumulation
for epoch in range(begin_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    loss_meter.reset()        

    ##Training_Start
    train_start_time = datetime.datetime.now()
    train(epoch)
    if (epoch % args.eval_period == 0 or epoch == args.nEpochs + 1):
        valid(args.scale)
    train_end_time = datetime.datetime.now()
    print('Epoch cost times: %s' % str(train_end_time-train_start_time))
    ##Training_End
    
    t = timer.t()
    prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
    
code_end_time = datetime.datetime.now()
print('Code cost times: %s' % str(code_end_time-code_start_time))

txt_write = open(args.validtext, 'a')
print('Code cost times: %s' % str(code_end_time-code_start_time))
txt_write.close()
##End_Total_Compuational_Time