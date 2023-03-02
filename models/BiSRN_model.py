from cProfile import label
import torch
import itertools
from .base_model import BaseModel
# from . import backbone
import torch.nn.functional as F
from BiSRN.BiSRNet import BiSRNet
from BiSRN.SSCDl import SSCDl
from BiSRN.losses import BiSRN_loss
from util.metrics import AverageMeter
from data.second_dataset import TensorIndex2Color
# baseline0 
class BiSRNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.istest = opt.istest
        if opt.phase == 'test':
            self.istest = True
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['seg','bn','sc','all']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['A', 'B', 'A_L_show','B_L_show', 'CD_L_show','pred_A_L_show','pred_B_L_show','pred_CD_show']  # visualizations for A and B
        if opt.phase == 'val':
            self.visual_names = ['semantic_A_L_show','semantic_B_L_show','pred_A_L_show','pred_B_L_show']
        if self.istest:
            self.visual_names = ['A', 'B','pred_A_L_show','pred_B_L_show','pred_CD_show']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['F']
        else:  # during test time, only load Gs
            self.model_names = ['F']
        # define networks (both Generators and discriminators)
        # self.netF = backbone.define_F(in_c=3, f_c=opt.f_c, type=opt.arch).to(self.device)
        if opt.arch == 'bisrn':
            self.netF = BiSRNet(num_classes=7).to(self.device)
        else:
            self.netF = SSCDl(num_classes=7).to(self.device)
        if self.isTrain:
            # define loss functions
            self.criterionF = BiSRN_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.SGD(filter(lambda p: p.requires_grad, self.netF.parameters()), lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
            # self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netF.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=5e-4)
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
        if not self.istest:
            self.A_L = input['A_L'].to(self.device)#[b,512,512]
            self.B_L = input['B_L'].to(self.device)
            self.CD_L = input['CD_L'].to(self.device)#[b,512,512]
            self.A_L_s = self.A_L.long()
            self.B_L_s = self.B_L.long()
            self.CD_L_s = self.CD_L.float()
            self.A_L_show = TensorIndex2Color(self.A_L_s).permute(0,3,1,2)#[b,3,512,512]
            self.B_L_show = TensorIndex2Color(self.B_L_s).permute(0,3,1,2)
            self.CD_L_show = self.CD_L_s.unsqueeze(1)#[b,1,512,512]

    def test(self, val=False):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            pred_A = self.preds_A.long()
            pred_B = self.preds_B.long()
            if val:  # score
                from util.metrics import RunningMetrics
                metrics = RunningMetrics(num_classes=7)

                metrics.update(self.A_L_s.detach().cpu().numpy(), pred_A.detach().cpu().numpy())
                metrics.update(self.B_L_s.detach().cpu().numpy(), pred_B.detach().cpu().numpy())
                scores = metrics.get_cm()
                return scores
            else:
                return pred_A, pred_B

    def val(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            preds_A = self.preds_A.cpu().detach().numpy()
            preds_B = self.preds_B.cpu().detach().numpy()
            labels_A = self.A_L_s.cpu().detach().numpy()
            labels_B = self.B_L_s.cpu().detach().numpy()
            return preds_A, preds_B, labels_A, labels_B

    def forward(self):
        self.cd, self.p1, self.p2 = self.netF(self.A, self.B)#[b,1,512,512],[b,7,512,512]
        change_mask = torch.sigmoid(self.cd)>0.5#[b,1,512,512]
        self.preds_A = torch.argmax(self.p1, dim=1)*change_mask.squeeze().long()
        self.preds_B = torch.argmax(self.p2, dim=1)*change_mask.squeeze().long()
        self.pred_CD_show = change_mask.long()
        self.pred_A_L_show = TensorIndex2Color(self.preds_A).permute(0,3,1,2)
        self.pred_B_L_show = TensorIndex2Color(self.preds_B).permute(0,3,1,2)
        semantic_A = torch.argmax(self.p1, dim=1).long()
        semantic_B = torch.argmax(self.p2, dim=1).long()
        self.semantic_A_L_show = TensorIndex2Color(semantic_A).permute(0,3,1,2)
        self.semantic_B_L_show = TensorIndex2Color(semantic_B).permute(0,3,1,2)
       
        # return self.pred_CD_show

    def backward(self):
        """Calculate the loss for generators F and L"""
        self.loss_seg, self.loss_bn, self.loss_sc = self.criterionF(self.cd, self.p1, self.p2, self.CD_L_s, self.A_L_s, self.B_L_s)

        self.loss_all = self.loss_seg + self.loss_bn + self.loss_sc
        if torch.isnan(self.loss_all):
           print(self.image_paths)

        self.loss_all.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute feat and dist

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
