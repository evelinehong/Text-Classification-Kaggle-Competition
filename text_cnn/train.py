#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn import metrics
from data_helpers import random_clip
from data_helpers import continuous_clip

def count(y):
	cnt = 0
	for i in range(len(y)):
		if y[i][1] > 0.5:
			cnt += 1
	print('0 vs 1: ' + str(len(y) - cnt) + ' vs ' + str(cnt))

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .025, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_title_file", "titleTrainInt.npy", "Data source for the training titles.")
tf.flags.DEFINE_string("train_text_file", "xTrainInt.npy", "Data source for the training texts.")
tf.flags.DEFINE_string("train_label_file", "yTrain.npy", "Data source for the training labels.")
tf.flags.DEFINE_string("word2vec_file", "word2vec.npy", "Data source for pretrained vectors.")
tf.flags.DEFINE_string("window_length", 1500, "Window length of model.")
tf.flags.DEFINE_integer("eval_num", 10, "Number of evaluation times (default: 1)")

# Model Hyperparameters
tf.flags.DEFINE_integer("title_length", 140, "Maximum length of input titles (default: 140)")
# tf.flags.DEFINE_integer("text_length", 2000, "Maximum length of input texts (default: 2000)")
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 400)")
tf.flags.DEFINE_string("filter_sizes", "1, 2, 3, 4, 5, 6, 7, 8, 9", "Comma-separated filter sizes (default: '1, 2, 3, 4, 5, 6, 7, 8, 9')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_convlayers", 1, "Number of conv layers (default: 1)")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 200)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularization lambda (default: 2.0)")

# Training parameters
tf.flags.DEFINE_integer("learning_rate", 0.01, "Learning rate (default: 0.001)")
tf.flags.DEFINE_integer("decay_rate", 0.97, "Decay rate (default: 0.97)")
tf.flags.DEFINE_integer("decay_every", 100, "Decay learning rate after this many steps (default: 100)")
tf.flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 80)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 500)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")
tf.flags.DEFINE_integer("num_checkpoints", 10000, "Number of checkpoints to store (default: 10000)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
FLAGS.batch_size = FLAGS.batch_size / FLAGS.eval_num * FLAGS.eval_num
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")

global total_p
global total_y
total_p = []
total_y = []

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
title, x, y = data_helpers.load_data_and_labels(FLAGS.train_title_file, FLAGS.train_text_file, FLAGS.train_label_file, FLAGS.title_length)

def repeat(data, times):
	new_data = []
	for x in data:
		for i in range(times):
			new_data.append(x)
	new_data = np.array(new_data)
	return new_data

def multiply(title, x, y):
	new_title = []
	new_x = []
	new_y = []
	for i in range(len(y)):
		if y[i][1] > 0.5:
			for j in range(11):
				new_title.append(title[i])
				new_x.append(x[i])
				new_y.append(y[i])
		else:
			new_title.append(title[i])
			new_x.append(x[i])
			new_y.append(y[i])
	new_title = np.array(new_title)
	new_x = np.array(new_x)
	new_y = np.array(new_y)
	return new_title, new_x, new_y

dictionary = np.load(FLAGS.word2vec_file)
print("Data loaded.")

# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
def shuffle(title, x, y):
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	title_shuffled = title[shuffle_indices]
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]
	return title_shuffled, x_shuffled, y_shuffled

title_shuffled, x_shuffled, y_shuffled = shuffle(title, x, y)

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
title_train, title_dev = title_shuffled[:dev_sample_index], title_shuffled[dev_sample_index:]
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

title_train, x_train, y_train = multiply(title_train, x_train, y_train)
title_dev, x_dev, y_dev = multiply(title_dev, x_dev, y_dev)

title_train, x_train, y_train = shuffle(title_train, x_train, y_train)
title_dev, x_dev, y_dev = shuffle(title_dev, x_dev, y_dev)
count(y_train)
count(y_dev)
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
	  allow_soft_placement=FLAGS.allow_soft_placement,
	  log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			title_length=FLAGS.title_length,
			sequence_length=FLAGS.window_length,
			num_classes=y_train.shape[1],
			dictionary=dictionary,
			embedding_size=FLAGS.embedding_dim,
			num_convlayers=FLAGS.num_convlayers,
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters=FLAGS.num_filters,
			l2_reg_lambda=FLAGS.l2_reg_lambda)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		learning_rate = tf.placeholder(tf.float32, shape=[])
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(update_ops):
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)

		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		# acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

		# Train Summaries
		train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		def train_step(title_batch, x_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
			  cnn.input_title: title_batch,
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
			  learning_rate : FLAGS.learning_rate,
			  cnn.is_training : True
			}
			
			_, step, summaries, loss, probabilities = sess.run(
				[train_op, global_step, train_summary_op, cnn.loss, cnn.probabilities],
				feed_dict)
			
			y_flat_batch = [label[0] < 0.5 and 1 or 0 for label in y_batch]
			fpr, tpr, thresholds = metrics.roc_curve(y_flat_batch, probabilities.transpose()[1])
			accuracy = metrics.auc(fpr, tpr)
			time_str = datetime.datetime.now().isoformat()
			# print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(title_batch, x_batch, y_batch, writer = None):
			"""
			Evaluates model on a dev set
			"""
			global total_y
			global total_p
			feed_dict = {
			  cnn.input_title: title_batch,
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: 1.0,
			  cnn.is_training : False
			}
			step, summaries, loss, probabilities = sess.run(
				[global_step, dev_summary_op, cnn.loss, cnn.probabilities],
				feed_dict)	
			y_flat_batch = [label[0] < 0.5 and 1 or 0 for label in y_batch]
			# fpr, tpr, thresholds = metrics.roc_curve(y_flat_batch, probabilities.transpose()[1])
			# accuracy = metrics.auc(fpr, tpr)
			time_str = datetime.datetime.now().isoformat()
			# print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			p = probabilities.transpose()[1]
			y = y_flat_batch
			p = p.reshape((-1, FLAGS.eval_num))
			p = np.mean(p, axis = 1).reshape(-1)
			y = np.reshape(y, (-1, FLAGS.eval_num))
			y = np.max(y, axis = 1).reshape(-1)
			total_p = np.concatenate((total_p, p))
			total_y = np.concatenate((total_y, y))
			if writer:
				writer.add_summary(summaries, step)
		
		# Generate batches
		for epoch in range(FLAGS.num_epochs):

			batches = data_helpers.batch_iter(list(zip(title_train, 
				continuous_clip(x_train, FLAGS.window_length, 1), y_train)), FLAGS.batch_size, 1)
				# random_clip(x_train, FLAGS.window_length), y_train)), FLAGS.batch_size, 1) notice this!!!!!!!!!
			# Training loop. For each batch...
			for batch in batches:
				title_batch, x_batch, y_batch = zip(*batch)
				train_step(title_batch, x_batch, y_batch)
				current_step = tf.train.global_step(sess, global_step)
				# print current_step
				if current_step % FLAGS.decay_every == 0:
					FLAGS.learning_rate = FLAGS.learning_rate * FLAGS.decay_rate	
				if current_step % FLAGS.evaluate_every == 0:
					# print("Evaluation:")
					dev_batches = data_helpers.batch_iter(list(zip(repeat(title_dev, FLAGS.eval_num), 
						continuous_clip(x_dev, FLAGS.window_length, FLAGS.eval_num),
						repeat(y_dev, FLAGS.eval_num))), FLAGS.batch_size, 1)
					total_y = []
					total_p = []
					for dev_batch in dev_batches:
						title_batch, x_batch, y_batch = zip(*dev_batch)
						dev_step(title_batch, x_batch, y_batch, writer=dev_summary_writer)
					fpr, tpr, thresholds = metrics.roc_curve(total_y, total_p)
					accuracy = metrics.auc(fpr, tpr)
					print("evaluate_acc on " + str(current_step) + ": " + str(accuracy) + " with learning_rate = " + str(FLAGS.learning_rate))
					# print("")
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=current_step)
					# print("Saved model checkpoint to {}".format(path))
		print ("done")
