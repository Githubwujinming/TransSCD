from .FCEF import FCEF
from .FCSiamConc import FCSiamConc
from .FCSiamDiff import FCSiamDiff
def getFCSModel(net='FCEF',in_c=3, nc=7):
    if net=='FCEF':
        print('using FCEF model')
        model = FCEF(in_c*2, nc=nc)
    elif net=='FCSC':
        print('using FCSiamConc model')
        model = FCSiamConc(in_c, nc=nc)
    elif net=='FCSD':
        print('using FCSiamDiff model')
        model = FCSiamDiff(in_c, nc=nc)
    # elif net=='str1':
    #     print("using HRSCD.str1 model")
    #     model = HRSCD1(in_dim=in_c, nc=nc)
    else:
        NotImplementedError('no such  FCS model!!')
    return model
