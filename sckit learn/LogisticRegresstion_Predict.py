from Tools import readbunchobj
from sklearn import preprocessing 
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier

trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)
print('1')

testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
print('2')

clf=XGBClassifier()
clf.fit(train_set.tdm, train_set.label)

predicted= clf.predict_proba(test_set.tdm)
predicted = predicted[:,1]

  
with open('predict_xgb2.csv','w',newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "pred"])
    for fid, expct_cate in zip(test_set.id, predicted):
        #print(fid)
        #print('1')
        writer.writerows([[fid,expct_cate]])
    # for expct in predicted:
    #     writer.writerow(expct)

print("预测完毕!!!")
