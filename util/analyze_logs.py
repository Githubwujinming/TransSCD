from os import path
from re import search


class DataExtractor(object):
    ''' DataExtrator class '''

    def __init__(self, infile, keyword, outfile=None):
        '''
        构造函数

        infile：输入文件名
        keyword：目标数据前面的关键字
        outfile：输出文件名
        '''

        self.infile = infile
        self.keyword = keyword
        self.outfile = outfile

    def data_after_keyword(self):
        ''' Extract data from infile after the keyword. '''

        try:
            data = []
            patt = '%s: (\d+\.?\d+)' % self.keyword  # 使用正则表达式搜索数据
            with open(self.infile, 'r') as fi:
                # with open(self.outfile, 'w') as fo:
                    for eachLine in fi:
                        s = search(patt, eachLine)
                        if s is not None:
                            # fo.write(s.group(1) + '\n')
                            data.append(float(s.group(1)))
            return data
        except IOError:
            print(
                "Open file [%s] or [%s] failed!" % (self.infile, self.outfile))
            return False
# def extrarct_logs(project):
#     val_infile = 'checkpoints/%s/val_log.txt'%project
#     val_keys = ['F_scd','Sek']
    
#     loss_infile = 'checkpoints/%s/loss_log.txt'%project
#     loss_keys = ['all']

#     val_extractors = [DataExtractor(val_infile, key) for key in val_keys]
#     loss_extractors = [DataExtractor(loss_infile, key) for key in loss_keys]
    