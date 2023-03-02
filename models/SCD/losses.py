import torch
import torch.nn.functional as F
from .metrics import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity, weighted_BCE

def SCD_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.8):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    # loss = loss_seg + loss_bn + loss_sc
    return (loss_seg, loss_bn, loss_sc)

def SCD2_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = F.cosine_similarity(outputs_A,outputs_B,dim=1)>thresold
    p_mask = label_unchange*p_mask
    labels_pa = torch.argmax(outputs_B,1)*p_mask
    labels_pb = torch.argmax(outputs_A,1)*p_mask
    s_mask = p_mask.unsqueeze(1).repeat(1,7,1,1)
    loss_pseg = criterion(outputs_A*s_mask, labels_pa) * 0.5 +  criterion(outputs_B*s_mask, labels_pb) * 0.5  
    return (loss_seg, loss_pseg, loss_sc)
    
def SCD3_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = label_unchange
    labels_pa = torch.argmax(outputs_B,1)*p_mask
    labels_pb = torch.argmax(outputs_A,1)*p_mask
    s_mask = p_mask.unsqueeze(1).repeat(1,7,1,1)
    loss_pseg = criterion(outputs_A*s_mask, labels_pa) * 0.5 +  criterion(outputs_B*s_mask, labels_pb) * 0.5  
    return (loss_seg, loss_pseg, loss_sc)
    
     
def SCD4_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    # loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = F.cosine_similarity(outputs_A,outputs_B,dim=1)>thresold
    p_mask = label_unchange*p_mask
    labels_pa = torch.argmax(outputs_B,1)*p_mask+labels_A
    labels_pb = torch.argmax(outputs_A,1)*p_mask+labels_B
    loss_pseg = criterion(outputs_A, labels_pa) * 0.5 +  criterion(outputs_B, labels_pb) * 0.5  
    return (loss_pseg, loss_bn, loss_sc)

def SCD5_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    # loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = label_unchange
    # print(torch.sum(p_mask * labels_A))
    labels_pa = torch.argmax(outputs_B,1)*p_mask+labels_A
    labels_pb = torch.argmax(outputs_A,1)*p_mask+labels_B
    loss_pseg = criterion(outputs_A, labels_pa) * 0.5 +  criterion(outputs_B, labels_pb) * 0.5  
    return (loss_pseg, loss_bn, loss_sc)

def SCD6_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    # loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = F.cosine_similarity(outputs_A,outputs_B,dim=1)>thresold
    p_mask = label_unchange&p_mask
    labels_A[p_mask] = torch.argmax(outputs_B,1)[p_mask]
    labels_B[p_mask] = torch.argmax(outputs_A,1)[p_mask]
    # labels_pa = torch.argmax(outputs_B,1)*p_mask+labels_A
    # labels_pb = torch.argmax(outputs_A,1)*p_mask+labels_B
    loss_pseg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_A) * 0.5  
    return (loss_pseg, loss_bn, loss_sc)


def SCD7_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.6):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    # loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    p_mask = label_unchange
    labels_A[p_mask] = torch.argmax(outputs_B,1)[p_mask]
    labels_B[p_mask] = torch.argmax(outputs_A,1)[p_mask]
    # labels_pa = torch.argmax(outputs_B,1)*p_mask+labels_A
    # labels_pb = torch.argmax(outputs_A,1)*p_mask+labels_B
    loss_pseg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_A) * 0.5  
    return (loss_pseg, loss_bn, loss_sc)

def SCD8_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B, thresold=0.8):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    # loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    
    label_unchange = ~labels_bn.bool()
    noise = torch.rand(labels_A.size()).to(labels_A.device)>thresold
    p_mask = label_unchange*noise
    labels_pa = torch.argmax(outputs_B,1)*p_mask+labels_A
    labels_pb = torch.argmax(outputs_A,1)*p_mask+labels_B
    loss_pseg = criterion(outputs_A, labels_pa) * 0.5 +  criterion(outputs_B, labels_pb) * 0.5  
    return (loss_pseg, loss_bn, loss_sc)