import os
import pickle
import csv
import json

from sklearn.datasets.base import Bunch
from Tools import readfile


def corpus2Bunch(wordbag_path, seg_path,csv_path):

    # 创建一个Bunch实例
    bunch = Bunch(label=[], id=[], contents=[])
    # 获取每个目录下所有的文件
    with open(seg_path,'rb') as f:
        with open(csv_path,'rt') as csvfile:
            reader=csv.DictReader(csvfile)
            for i,rows in enumerate(reader):
                #if i==64409:
                    #break
                #else:
                    bunch.contents.append(f.readline())
                    bunch.label.append(rows['pred'])
                    bunch.id.append(rows['id'])

    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建训练集文本对象结束！！！")

def corpus2Bunch4test(wordbag_path, seg_path,json_file):

    # 创建一个Bunch实例
    bunch = Bunch(label=[], id=[], contents=[])
    # 获取每个目录下所有的文件
    i=0
    with open(json_file,'r') as jf:
        with open(seg_path,'rb') as f:
            for line,idline in zip(f,jf):
                bunch.contents.append(line)#用f.readline()会出现数据集个数变少，仅2w+
                dic = json.loads(idline)
                bunch.id.append(dic['id'])
                print(i)
                i=i+1
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建测试集文本对象结束！！！")



if __name__ == "__main__":
    #对训练集进行Bunch化操作：
    #wordbag_path = "train_word_bag1/train_set.dat"  # Bunch存储路径
    wordbag_path = "test_set.dat"
    seg_path = "seg1.txt"  # 分词后分类语料库路径
    json_file = "test.json"
    corpus2Bunch4test(wordbag_path, seg_path, json_file)
    #已完成

    #对测试集进行Bunch化操作：
    #wordbag_path = "test_word_bag1/test_set.dat"  # Bunch存储路径
    #seg_path ="seg_test.txt"  # 分词后分类语料库路径
    #corpus2Bunch4test(wordbag_path, seg_path,'test.json')