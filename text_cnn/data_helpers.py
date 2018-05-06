import numpy as np
import copy
from gensim.models import Word2Vec
embedding = Word2Vec.load('Word2Vec/wiki.zh.text.model')

def random_clip(raw_data, window_length):
	data = np.zeros((len(raw_data), window_length))
	for i in range(len(raw_data)):
		if len(raw_data[i]) < window_length:
			newdata = np.zeros(window_length)
			for j in range(len(raw_data[i])):
				newdata[j] = raw_data[i][j]
			data[i] = newdata
		else:
			start_index = np.random.randint(len(raw_data[i]) - window_length + 1)
			data[i] = np.array(raw_data[i][int(start_index) : int(start_index + window_length)])
	return data

def continuous_clip(data, window_length, times):
	new_data = np.zeros((len(data) * times, window_length))
	cnt = 0
	for i in range(len(data)):
		if len(data[i]) < window_length or times == 1:
			newdata = np.zeros(window_length)
			for j in range(min(len(data[i]), window_length)):
				newdata[j] = data[i][j]
			for j in range(times):
				new_data[cnt] = newdata
				cnt += 1
		else:
			right = len(data[i]) - window_length
			step = right / (times - 1)
			for j in range(times):
				head = j * step
				if j == times - 1:
					head = right
				new_data[cnt] = data[i][int(head) : int(head + window_length)]
				cnt += 1
	return new_data

def segment(length, window, num):
	right = length - (window - 1)
	step = right / (num - 1)
	for i in range(num):
		head = i * step
		if i == num - 1:
			head = right - 1
		print (head, head + window - 1)

def truncate(raw, length):
	proc = np.zeros((raw.shape[0], length), dtype=int)
	cnt1 = 0
	for x in raw:
		cnt2 = 0
		for word in x:
			if cnt2 >= length:
				break
			proc[cnt1][cnt2] = raw[cnt1][cnt2]
			cnt2 += 1
		cnt1 += 1
	return proc

def load_data_and_labels(title_file, text_file, label_file, title_length):
	title_raw = np.load(title_file)
	x_raw = np.load(text_file)
	y_raw = np.load(label_file)
	title = truncate(title_raw, title_length)
	x_text = np.array(x_raw)
	y = []
	for label in y_raw:
		label = label.decode()
		if label != '0' and label != '1':
			raise Exception('Wrong Label: ' + label)
		if int(label) == 0:
			y.append([1, 0])
		else:
			y.append([0, 1])
	y = np.array(y)
	return [title, x_text, y]

def load_data_and_ids(title_file, text_file, id_file, title_length):
	title_raw = np.load(title_file)
	x_raw = np.load(text_file)
	id_raw = np.load(id_file)
	Id = np.array(id_raw)
	title = truncate(title_raw, title_length)
	x_text = np.array(x_raw)
	return [title, x_text, Id]

def batch_iter(data, batch_size, num_epochs, shuffle=True, test=False):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			raw_data = shuffled_data[int(start_index):int(end_index)]
			yield raw_data
			'''
			if not test:
				yield random_clip(raw_data, window_length)
			else:
				yield continuous_clip(raw_data, window_length, times)
			# yield process(raw_data, text_length, embedding_dim, test)
			'''
