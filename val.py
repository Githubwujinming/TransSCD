import time

import torch
from options.test_options import TestOptions
from data import create_dataset
from data.second_dataset import num_classes
from models import create_model
import os
from util.util import save_visuals
from util.metrics import AverageMeter, accuracy, SCDD_eval_all
import numpy as np
from util.util import mkdir

def make_val_opt(opt):

    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt.no_flip2 = True    # no flip; comment this line if results on flipped images are needed.

    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'val'
    opt.preprocess = ''
    opt.isTrain = False
    opt.aspect_ratio = 1
    opt.eval = True


    opt.angle = 0
    opt.results_dir = './results/'

    opt.num_test = np.inf
    return opt


def print_current_acc(log_name, epoch, score):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """

    message = '(epoch: %s) ' % str(epoch)
    for k, v in score.items():
        message += '%s: %.4f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def val(opt):

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    save_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    mkdir(save_path)
    model.eval()
    # create a logging file to store training losses
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val1_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        score = model.test(val=True)           # run inference return confusion_matrix
        running_metrics.update(score)

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_visuals(visuals,save_path,img_path[0])
    score = running_metrics.get_scores()
    print_current_acc(log_name, opt.epoch, score)



def val2(opt):

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # model.netF.load_state_dict(torch.load('checkpoints/SECOND_BiSRN_1001/BiSRN_orginal_net_F.pth'), strict=False)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    save_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    mkdir(save_path)
    model.eval()
    # create a logging file to store training losses
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val1_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        preds_A, preds_B, labels_A, labels_B = model.val()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            acc_meter.update(acc)

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_visuals(visuals,save_path,img_path[0])
    score = SCDD_eval_all(preds_all, labels_all,num_classes)
    score.update({'OA':acc_meter.average()*100})
    print_current_acc(log_name, opt.epoch, score)


def make_extra_opt(opt):
    opt.cfg='configs/SCD3_Csa.yaml'
    opt.prob=0
    return opt
# SSCDl+SCLoss_48e_mIoU73.12_Sek22.63_score37.78_OA87.68
if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    opt = make_val_opt(opt)
    opt = make_extra_opt(opt)
    opt.phase = 'val'
    opt.dataroot = "/data/datasets/SECOND_scd/test"

    # opt.dataset_mode = 'RSST'
    opt.dataset_mode = 'second'

    opt.n_class = 7
    opt.num_threads=1
    
    opt.arch = 'str4'
    opt.model = 'SCD'
    opt.gpu_ids=[1]
    opt.name = 'SECOND_'+'SCD3_csa_0210'
    opt.results_dir = 'SECOND_'+'SCD3_csa_0210'

    # opt.epoch = 'orginal'
    # opt.epoch = '42_F_scd_59.93840'
    # opt.ds=8
    opt.num_test = np.inf

    val(opt)