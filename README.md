# Text-Classification-Kaggle-Competition
This is my EE448 project, in which I ranked 2nd in a kaggle competition.<br>
It is a Chinese text classification competition.<br>
这是一个中文文本分词比赛。<br>
https://www.kaggle.com/c/ee448-2018-text-classification<br>

本次比赛用三个模型：CNN, logistic regression和xgboost。<br>
其中CNN为主力，和xgboost ensemble后有极大地提升。logistic regression的ensemble帮助不大。<br>
对于CNN，参数扰动的结果取均值。<br>
最后是50多个csv用rank加权或1/rank加权得出的结果。<br>
除了CNN外用的都是sklearn的简单代码。可参考https://blog.csdn.net/github_36326955/article/details/54891204。<br>

数据下载：https://jbox.sjtu.edu.cn/link/view/aae23aff5b4e4bcb866158642edeaf6b
