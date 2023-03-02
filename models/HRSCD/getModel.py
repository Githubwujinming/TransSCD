# from .FCEF_Res import *
from .str3 import HRSCD3
from .str4 import HRSCD4
def getHRSCDModel(net='str1',in_c=3, nc=7):
    if net=='str3':
        print('using str3 model')
        model = HRSCD3(in_c, nc)
    elif net=='str4':
        print('using str4 model')
        model = HRSCD4(in_c, nc)
    else:
        NotImplementedError('no such  FCS model!!')
    return model
