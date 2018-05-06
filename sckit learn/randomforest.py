import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV  
from sklearn import cross_validation, metrics 
import pandas as pd
import matplotlib.pylab as plt  
from Tools import readbunchobj
import csv

trainpath = "chinese/train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)
print('1')

testpath = "chinese/test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)
print('2')

Label=[int(x) for x in train_set.label]

# rf0 = RandomForestClassifier(oob_score=True, random_state=10)  
# rf0.fit(train_set.tdm, Label)  
# print (rf0.oob_score_ )
# y_predprob = rf0.predict_proba(train_set.tdm)[:,1] 
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(Label,y_predprob))

gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=50,  min_samples_leaf=20,max_depth=8,max_features='sqrt' , n_estimators = 500, random_state=10),    scoring='roc_auc',cv=5, param_grid ={'max_depth':[50,80,2]})  
gsearch1.fit(train_set.tdm, Label)  
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
