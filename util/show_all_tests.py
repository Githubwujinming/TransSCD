
import os
import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
from .html import HTML
from .util import mkdir
class HTMLCOMA(HTML):
    def __init__(self, web_dir, title, refresh=0):
        HTML.__init__(self, web_dir, title, refresh=refresh)
    
    def add_images(self, ims, txts, links, width=200):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)
                            
    def save(self, filename='index'):
        """save the current content to the HMTL file"""
        html_file = '%s/%s.html' % (self.web_dir, filename)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

# 把一个数据集的各个模型测试结果保存到一个html页面中
def cd():
    dataset = ['GD','LEVIR','SYSU','SV']
    methods = ['ChangeFormer','DSAMN','DSIFN','FCCDN','SNUN','CUBE']
    # methods = ['CUBE']
    txts = ['A','B','L']
    for m in methods:
        txts.append('%s_result'%m)
    web_dir = 'web_results'
    As = []
    Bs = []
    Ls = []
    comps = {}
    ind = 0
    for i in range(len(methods)):
        data = dataset[ind]
        dir = os.path.join(web_dir, data)
        imgs_name = sorted(os.listdir(dir+'/%s_%s'%(data,methods[i])))
        if len(As) == 0:
            As = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'A.png' in a]
            Bs = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'B.png' in a]
            Ls = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'L.png' in a]
        comps[methods[i]] = [os.path.join('%s_%s'%(data,methods[i]),a) for a in imgs_name if 'comp.png' in a]
    html = HTMLCOMA(os.path.join(web_dir,dataset[ind]),'%s_tests_results.html'%dataset[ind])
    html.add_header('%s_tests_results.html'%dataset[ind])
    for i in range(len(As)):
        ims = []
        A = As[i]
        B = Bs[i]
        L = Ls[i]
        comp = [comps[m][i] for m in methods]
        ims.extend((A,B,L))
        ims.extend(comp)
        txts[0] = A.split('.png')[0].split('/')[1]
        html.add_images(ims,txts,ims)
    html.save()
'''
使用方法
在comp_dirs_list中指定要比较方法的测试结果目录，注意要在test时
batch size要一致，这样每个方法的图片才是对应的
'''
def scd_show(comp_dirs_list,save_name):
    # comp_dirs_list = ['checkpoints/SECOND_BISRN_1129/val_latest','checkpoints/SECOND_HRSCD3_1214/val_latest']
    methods = [a.split('/')[-2] for a in comp_dirs_list]
    # comp_dirs_list = [os.path.join(a, 'images') for a in comp_dirs_list]
    txtA = ['A','A_show']
    txtB = ['B','B_show']
    for m in methods:
        txtA.append(m)
        txtB.append(m)
    As = []
    Bs = []
    As_show = []
    Bs_show = []
    compsA = {}
    compsB = {}
    for i in range(len(methods)):
        dir = comp_dirs_list[i]
        imgs_name = sorted(os.listdir(dir))
        if len(As) == 0:
            As = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'A.png' in a]
            Bs = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'B.png' in a]
            As_show = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'A_L_show.png' in a and 'pred' not in a]
            Bs_show = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'B_L_show.png' in a and 'pred' not in a]
        compsA[methods[i]] = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'pred_A' in a]
        compsB[methods[i]] = [os.path.join('../', comp_dirs_list[i], a) for a in imgs_name if 'pred_B' in a]
    
    web_dir = 'web_results'
    html = HTMLCOMA(os.path.join(web_dir),'tests_results.html')
    html.add_header('tests_results.html')
    for i in range(len(As)):
        imsA = []
        imsB = []
        A = As[i]
        B = Bs[i]
        A_show = As_show[i]
        B_show = Bs_show[i]
        compA = [compsA[m][i] for m in methods]
        compB = [compsB[m][i] for m in methods]
        imsA.extend((A,A_show))
        imsA.extend(compA)
        imsB.extend((B,B_show))
        imsB.extend(compB)
        txtA[0] = A.split('.png')[0].split('/')[-1]
        txtB[0] = B.split('.png')[0].split('/')[-1]
        html.add_images(imsA,txtA,imsA)
        html.add_images(imsB,txtB,imsB)
    html.save(save_name)
'''
将所有方法的预测目录放到同一父目录下，methods指定对比方法
'''
def scd_show2(save_name='test_index',methods=['TransSCD','BiSRN','str3','str4','DSAHRN']):
    # comp_dirs_list = ['checkpoints/SECOND_BISRN_1129/val_latest','checkpoints/SECOND_HRSCD3_1214/val_latest']
    # methods 
    # comp_dirs_list = [os.path.join(a, 'images') for a in comp_dirs_list]
    txtA = ['A','A_show']
    txtB = ['B','B_show']
    # txtC = ['A','A_show']
    # txtD = ['B','B_show']
    for m in methods:
        txtA.append(m)
        txtB.append(m)
        # txtC.append(m)
        # txtD.append(m)
    As = []
    Bs = []
    As_show = []
    Bs_show = []
    compsA = {}
    compsB = {}
    # comps_semA = {}
    # comps_semB = {}
    web_dir = 'web_results'
    
    for i in range(len(methods)):
        dir = os.path.join(web_dir,methods[i])
        imgs_name = sorted(os.listdir(dir))
        if len(As) == 0:
            As = [os.path.join(methods[i], a) for a in imgs_name if 'A.png' in a]
            Bs = [os.path.join(methods[i], a) for a in imgs_name if 'B.png' in a]
            As_show = [os.path.join(methods[i], a) for a in imgs_name if 'A_L_show.png' in a and 'pred' not in a and 'semantic_A' not in a]
            Bs_show = [os.path.join(methods[i], a) for a in imgs_name if 'B_L_show.png' in a and 'pred' not in a and 'semantic_B' not in a]
        compsA[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'pred_A' in a]
        compsB[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'pred_B' in a]
        # comps_semA[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'semantic_A_L' in a]
        # comps_semB[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'semantic_B_L' in a]
    
    html = HTMLCOMA(os.path.join(web_dir),'tests_results.html')
    html.add_header('tests_results.html')
    for i in range(len(As)):
        imsA = []
        imsB = []
        imsC = []
        imsD = []
        A = As[i]
        B = Bs[i]
        A_show = As_show[i]
        B_show = Bs_show[i]
        compA = [compsA[m][i] for m in methods]
        compB = [compsB[m][i] for m in methods]
        # semA = [comps_semA[m][i] for m in methods]
        # semB = [comps_semB[m][i] for m in methods]
        imsA.extend((A,A_show))
        # imsC.extend((A,A_show))
        imsA.extend(compA)
        # imsC.extend(semA)
        imsB.extend((B,B_show))
        # imsD.extend((B,B_show))
        imsB.extend(compB)
        # imsD.extend(semB)
        
        txtA[0] = A.split('.png')[0].split('/')[-1]
        txtB[0] = B.split('.png')[0].split('/')[-1]
        html.add_images(imsA,txtA,imsA)
        html.add_images(imsB,txtB,imsB)
        # html.add_images(imsC,txtC,imsC)
        # html.add_images(imsD,txtD,imsD)
    html.save(save_name)
    
        
    
def semantic_show(save_name='test_index',methods=['TransSCD','BiSRN','str3','str4','DSAHRN']):
    # comp_dirs_list = ['checkpoints/SECOND_BISRN_1129/val_latest','checkpoints/SECOND_HRSCD3_1214/val_latest']
    # methods 
    # comp_dirs_list = [os.path.join(a, 'images') for a in comp_dirs_list]
    # txtA = ['A','A_show']
    # txtB = ['B','B_show']
    txtC = ['A','A_show']
    txtD = ['B','B_show']
    for m in methods:
        # txtA.append(m)
        # txtB.append(m)
        txtC.append(m)
        txtD.append(m)
    As = []
    Bs = []
    As_show = []
    Bs_show = []
    # compsA = {}
    # compsB = {}
    comps_semA = {}
    comps_semB = {}
    web_dir = 'web_results'
    
    for i in range(len(methods)):
        dir = os.path.join(web_dir,methods[i])
        imgs_name = sorted(os.listdir(dir))
        if len(As) == 0:
            As = [os.path.join(methods[i], a) for a in imgs_name if 'A.png' in a]
            Bs = [os.path.join(methods[i], a) for a in imgs_name if 'B.png' in a]
            As_show = [os.path.join(methods[i], a) for a in imgs_name if 'A_L_show.png' in a and 'pred' not in a and 'semantic_A' not in a]
            Bs_show = [os.path.join(methods[i], a) for a in imgs_name if 'B_L_show.png' in a and 'pred' not in a and 'semantic_B' not in a]
        # compsA[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'pred_A' in a]
        # compsB[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'pred_B' in a]
        comps_semA[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'semantic_A_L' in a]
        comps_semB[methods[i]] = [os.path.join(methods[i], a) for a in imgs_name if 'semantic_B_L' in a]
    
    html = HTMLCOMA(os.path.join(web_dir),'tests_results.html')
    html.add_header('tests_results.html')
    for i in range(len(As)):
        # imsA = []
        # imsB = []
        imsC = []
        imsD = []
        A = As[i]
        B = Bs[i]
        A_show = As_show[i]
        B_show = Bs_show[i]
        # compA = [compsA[m][i] for m in methods]
        # compB = [compsB[m][i] for m in methods]
        semA = [comps_semA[m][i] for m in methods]
        semB = [comps_semB[m][i] for m in methods]
        # imsA.extend((A,A_show))
        imsC.extend((A,A_show))
        # imsA.extend(compA)
        imsC.extend(semA)
        # imsB.extend((B,B_show))
        imsD.extend((B,B_show))
        # imsB.extend(compB)
        imsD.extend(semB)
        
        txtC[0] = A.split('.png')[0].split('/')[-1]
        txtD[0] = B.split('.png')[0].split('/')[-1]
        # html.add_images(imsA,txtA,imsA)
        # html.add_images(imsB,txtB,imsB)
        html.add_images(imsC,txtC,imsC)
        html.add_images(imsD,txtD,imsD)
    html.save(save_name)
    
if __name__ == '__main__':
    
    scd_show()