#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from data_helpers import continuous_clip
import csv

def repeat(data, times):
	new_data = []
	for x in data:
		for i in range(times):
			new_data.append(x)
	new_data = np.array(new_data)
	return new_data

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_title_file", "titleTestInt.npy", "Data source for the test titles.")
tf.flags.DEFINE_string("test_text_file", "xTestInt.npy", "Data source for the test texts.")
tf.flags.DEFINE_string("test_id_file", "idTest.npy", "Data source for the test ids.")
tf.flags.DEFINE_integer("title_length", 140, "Maximum length of input titles (default: 140)")
tf.flags.DEFINE_integer("window_length", 1500, "Length of window (default: 2500)")
# tf.flags.DEFINE_integer("text_length", 1000, "Maximum length of input texts (default: 1000)")
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 400)")
tf.flags.DEFINE_integer("eval_num", 10, "Number of evaluation times (default: 1)")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 80)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1523795518/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_file", "model-500", "Checkpoint file from training run")

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

# CHANGE THIS: Load data. Load your own data here
title_test, x_test, id_test = data_helpers.load_data_and_ids(FLAGS.test_title_file, FLAGS.test_text_file, FLAGS.test_id_file, FLAGS.title_length)

print("\nTesting...\n")

# Testion
# ==================================================
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
	  allow_soft_placement=FLAGS.allow_soft_placement,
	  log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir+"{}.meta".format(FLAGS.checkpoint_file))
		saver.restore(sess, FLAGS.checkpoint_dir+FLAGS.checkpoint_file)

		# Get the placeholders from the graph by name
		input_title = graph.get_operation_by_name("input_title").outputs[0]
		input_x = graph.get_operation_by_name("input_x").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		is_training = graph.get_operation_by_name("is_training").outputs[0]

		# Tensors we want to evaluate
		probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]

		# Generate batches for one epoch
		batches = data_helpers.batch_iter(list(zip(repeat(title_test, FLAGS.eval_num), 
			continuous_clip(x_test, FLAGS.window_length, FLAGS.eval_num))), FLAGS.batch_size, 1, shuffle=False, test=True)

		# Collect the predictions here
		all_probabilities = []

		for batch in batches:
			title_test_batch, x_test_batch = zip(*batch)
			batch_probabilities = sess.run(probabilities, {input_title: title_test_batch, input_x: x_test_batch, dropout_keep_prob: 1.0, is_training: False})
			prob = batch_probabilities.transpose()[1].reshape(-1, FLAGS.eval_num)
			prob = np.mean(prob, axis = 1).reshape(-1)
			all_probabilities = np.concatenate((all_probabilities, prob))
			
# Save the evaluation to a csv

print (np.array(id_test).shape)
print (all_probabilities.shape)

probabilities_human_readable = np.column_stack((np.array(id_test), all_probabilities))
out_path = os.path.join("probabilities.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
	csv.writer(f).writerows([['id', 'pred']])
	csv.writer(f).writerows(probabilities_human_readable)

