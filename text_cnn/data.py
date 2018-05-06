# coding:utf-8
import json
# import jieba
import numpy as np
import sys
import csv

if __name__ == '__main__':
    #jieba.enable_parallel(4)
    # stopwords = [line.strip() for line in open('stopwords2.txt', 'r', encoding='utf-8').readlines()]
    label1 = []
    yTrain = []
    filename = 'train.csv'
    with open(filename) as f2:
        reader = csv.reader(f2)
        for row in reader:
            if row[1] == '0' or row[1] == '1':
                yTrain.append(row[1])
    # #ain = []
    yTrain = []
    titleTrain = []
    idTrain = []
    with open('title.txt', 'ab') as fW:
    with open('train.json', 'r', encoding='utf-8') as f:
        cnt1 = 0
        for line in f:
            cnt1+=1
            if cnt1 % 1000 ==0:
                print(cnt1)
            # # line1 = line.replace('html>','')
            # line2 = line1.replace('body>','')
            line2 = line.replace('<title>','')
            # line3 = line2.replace('title>','\n')
            # line4 = line3.replace('<//html>','')
            # line5 = line4.replace('<//body>','')
            # line6 = line5.replace('<//title>','\n')
            dic = json.loads(line2)
            fileTrainRead1 = dic ['title']
            fileTrainSeg1 = []
            title = []
            seg_title=jieba.cut(fileTrainRead1, cut_all=False)
            for item in seg_title:
                item=item.strip();
                if item not in stopwords:
                    title.append(item)
            titleTrain.append(title) 

            # fileTrainRead2 = dic ['content']
            # fileTrainSeg2 = []
            # content = []
            # seg_content=jieba.cut(fileTrainRead2, cut_all=False)
            # for item in seg_content:
            #     item=item.strip();
            #     if item not in stopwords:
            #         content.append(item)
            # xTrain.append(content)

          # Id = dic ['id']
            # idTrain.append(id)

          label = label1[cnt1]
            yTrain.append(label)
            
          # with open('train.csv') as f:
          #     reader = csv.reader(f)
          #     for row in reader:
          # filename = 'train.csv'
          # with open(filename) as f2:
          #     reader = csv.reader(f2)
          #     for i,rows in enumerate(reader):
          #         if i == cnt1:
          #             label = rows[1]
          #             yTrain.append(label)

    np.save('xTrain', xTrain)
    np.save('yTrain', yTrain)
    np.save('idTrain', idTrain)
    np.save('titleTrain', titleTrain)
    

                #         fW.write(item.encode('utf-8'))
                #         fW.write(' '.encode('utf-8'))
                # fW.write('\n'.encode('utf-8'))

                # fileTrainRead= dic ['content']
                # fileTrainSeg = []
                # seg_content=jieba.cut(fileTrainRead, cut_all=False)
                # for item in seg_content:
                #     item=item.strip();
                #     if item not in stopwords:
                #         fW.write(item.encode('utf-8'))
                #         fW.write(' '.encode('utf-8'))
                # fW.write('\n'.encode('utf-8'))
                #print('1')