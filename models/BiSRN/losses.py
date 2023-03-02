from .metrics import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity, weighted_BCE

def BiSRN_loss(out_change, outputs_A, outputs_B, labels_bn, labels_A, labels_B):
    criterion = CrossEntropyLoss2d(ignore_index=0)
    criterion_sc = ChangeSimilarity()
    loss_seg = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
    loss_bn = weighted_BCE_logits(out_change, labels_bn)
    loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
    # loss = loss_seg + loss_bn + loss_sc
    return (loss_seg, loss_bn, loss_sc)