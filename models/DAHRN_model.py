
import torch
import itertools

from DAHRN.resnet import BasicBlock, Bottleneck
from .base_model import BaseModel
from BiSRN.losses import BiSRN_loss
import torch.nn.functional as F
# from DAHRN.losses import DCAN_loss
from DAHRN.CDNet import  BASECDNet
from data.second_dataset import TensorIndex2Color
from DAHRN.VAN import Block
# baseline0 
BLOCK = {
    'block':BasicBlock,
    'resnet':BasicBlock,
    'lka':Block,
    'Bottleneck':Bottleneck,
    'lkablock':Block
}

HRBLOCKS = {
    'tiny': [1,1,1,1],
    'small': [1,1,2,2],
    'base':[2,2,2,2],
    'large': [3,3,3,3]
}
class DAHRNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('-b','--Block',type=str, default='BasicBlock', help='which type block you want to use in decoder')
        # parser.add_argument('-e','--encoder', type=str, default='vtiny', help='which encoder you want to use')
        # # parser.add_argument('-c','--cube_num', type=int, default=3, help='cube num in decoder')
        # parser.add_argument('-r','--hrblocks', type=str, default='base', help='num_blocks in hrmodule')
        # # parser.add_argument('-d','--depth', type=int, default=4, help='branch number for CD')
        # parser.add_argument('--bilinear', type=bool, default=True, help='wether use bilinar to upsample or not,if False,than us transpose convolution')
        # parser.add_argument('-f','--fuse', type=bool, default=False, help='fuse or not in cube')
        # parser.add_argument('-o','--output_stride',type=int, default=32,help='times to narrow compare with input')
        # parser.add_argument('-l','--criterion', type=str, default='mfpnloss', help='which loss function you want to use, detailed in loss.py')
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
        # self.criterion_type=opt.criterion
        block = BLOCK['block'] 
        # block = BLOCK[opt.Block] if opt.Block in BLOCK.keys() else BasicBlock
        num_blocks = HRBLOCKS['base']
        self.netF = BASECDNet(backbone='hrnet18',block=block,num_blocks=num_blocks,cube_num=1, 
                            num_branches=4).to(self.device)
        if self.isTrain:
            # define loss functions
            self.criterionF = BiSRN_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netF.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
            self.A_L = input['A_L'].to(self.device)
            self.B_L = input['B_L'].to(self.device)
            self.CD_L = input['CD_L'].to(self.device)
            self.A_L_s = self.A_L.long()
            self.B_L_s = self.B_L.long()
            self.CD_L_s = self.CD_L.float()
            self.A_L_show = TensorIndex2Color(self.A_L_s).permute(0,3,1,2)
            self.B_L_show = TensorIndex2Color(self.B_L_s).permute(0,3,1,2)
            self.CD_L_show = self.CD_L_s.unsqueeze(1)

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


    def forward(self):
        self.p1, self.p2, self.cd = self.netF(self.A, self.B)
        self.p1 = F.interpolate(self.p1, size=self.A.shape[2:], mode='bilinear',align_corners=True)
        self.p2 = F.interpolate(self.p2, size=self.B.shape[2:], mode='bilinear',align_corners=True)
        self.cd = F.interpolate(self.cd, size=self.B.shape[2:], mode='bilinear',align_corners=True)
        change_mask = torch.sigmoid(self.cd)>0.5
        self.preds_A = torch.argmax(self.p1, dim=1)*change_mask.squeeze().long()
        self.preds_B = torch.argmax(self.p2, dim=1)*change_mask.squeeze().long()
        self.pred_CD_show = change_mask.long()
        self.pred_A_L_show = TensorIndex2Color(self.preds_A).permute(0,3,1,2)
        self.pred_B_L_show = TensorIndex2Color(self.preds_B).permute(0,3,1,2)
        semantic_A = torch.argmax(self.p1, dim=1).long()
        semantic_B = torch.argmax(self.p2, dim=1).long()
        self.semantic_A_L_show = TensorIndex2Color(semantic_A).permute(0,3,1,2)
        self.semantic_B_L_show = TensorIndex2Color(semantic_B).permute(0,3,1,2)
       

    def backward(self):
        """Calculate the loss for generators F and L"""
        # print(self.p1.shape, self.p2.shape, self.cd.shape)
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
