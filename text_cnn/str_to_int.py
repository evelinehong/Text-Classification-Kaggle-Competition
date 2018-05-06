import numpy as np
from gensim.models import Word2Vec
embedding = Word2Vec.load('weibo/word2vec')

xTrain = np.load('xTrain.npy')
xTest = np.load('xTest.npy')
titleTrain = np.load('titleTrain.npy')
titleTest = np.load('titleTest.npy')
cnt = 0
dic = {}

word2vec = []

xTrainInt = []
for text in xTrain:
	textInt = []
	for word in text:
		vec = None
		try:
			vec = embedding.wv[word]
		except:
			pass
		if vec is None:
			continue
		if dic.get(word) is None:
			dic[word] = cnt
			word2vec.append(vec)
			cnt += 1
		textInt.append(dic[word])
	xTrainInt.append(textInt)

np.save('xTrainInt', xTrainInt)

xTestInt = []
for text in xTest:
	textInt = []
	for word in text:
		vec = None
		try:
			vec = embedding.wv[word]
		except:
			pass
		if vec is None:
			continue
		if dic.get(word) is None:
			dic[word] = cnt
			word2vec.append(vec)
			cnt += 1
		textInt.append(dic[word])
	xTestInt.append(textInt)

np.save('xTestInt', xTestInt)

titleTrainInt = []
for text in titleTrain:
	textInt = []
	for word in text:
		vec = None
		try:
			vec = embedding.wv[word]
		except:
			pass
		if vec is None:
			continue
		if dic.get(word) is None:
			dic[word] = cnt
			word2vec.append(vec)
			cnt += 1
		textInt.append(dic[word])
	titleTrainInt.append(textInt)
np.save('titleTrainInt', titleTrainInt)	

titleTestInt = []
for text in titleTest:
	textInt = []
	for word in text:
		vec = None
		try:
			vec = embedding.wv[word]
		except:
			pass
		if vec is None:
			continue
		if dic.get(word) is None:
			dic[word] = cnt
			word2vec.append(vec)
			cnt += 1
		textInt.append(dic[word])
	titleTestInt.append(textInt)
np.save('titleTestInt', titleTestInt)

np.save('word2vec', word2vec)

