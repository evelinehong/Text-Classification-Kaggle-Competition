import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None):
	shape = input_.get_shape().as_list()
	if len(shape) != 2:
		raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	if not shape[1]:
		raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	input_size = shape[1]
	with tf.variable_scope(scope or "SimpleLinear"):
		matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
		bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
	return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, layer_size=1, bias=0, f=tf.nn.relu):
	output = input_
	for idx in range(layer_size):
		output = f(linear(output, size, scope='output_lin_%d' % idx))
		transform_gate = tf.sigmoid(linear(input_, size, scope='transform_lin_%d' % idx) + bias)
		carry_gate = 1. - transform_gate
		output = transform_gate * output + carry_gate * input_
	return output

class TextCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
	  self, title_length, sequence_length, num_classes, dictionary, num_convlayers,
	  embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# Placeholders for input, output and dropout
		self.input_title = tf.placeholder(tf.int32, [None, title_length], name="input_title")
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		#self.input = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input = tf.concat(1,[self.input_title, self.input_x])
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.is_training = tf.placeholder(tf.bool, name='is_training')

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			# self.embedded_titles = tf.nn.embedding_lookup(dictionary, self.input_title)
			self.embedded_chars = tf.nn.embedding_lookup(dictionary, self.input)
			# self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

		# Create a convolution + maxpool layer for each filter size
		self.h_pool = self.embedded_chars
		num_filters_total = num_filters * len(filter_sizes)
		
		pooling_size = int(round(pow(sequence_length, 1. / num_convlayers)))
		length = title_length + sequence_length
		for layer_num in range(num_convlayers):
			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s-%s" % (layer_num, filter_size)):
					# Convolution Layer
					if layer_num == 0:
						filter_shape = [filter_size, embedding_size, num_filters]
					else:
						filter_shape = [filter_size, num_filters_total, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
					conv = tf.nn.conv1d(
						self.h_pool,
						W,
						stride=1,
						padding="SAME",
						name="conv")
					conv = tf.nn.bias_add(conv, b)
					batch_norm = conv # tf.contrib.layers.batch_norm(conv, decay = 0.99, center = True, scale = True, epsilon = 1e-7, is_training = self.is_training, scope = 'bn')
					# Apply nonlinearity
					h = tf.nn.relu(batch_norm, name="relu")
					h = tf.reshape(h, [-1, length, num_filters, 1])
					# Maxpooling over the outputs
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, pooling_size, 1, 1],
						strides=[1, pooling_size, 1, 1],
						padding='VALID',
						name="pool")
					pooled_outputs.append(pooled)

			# Combine all the pooled features
			length /= pooling_size
			length = int(length)
			self.h_pool = tf.concat(2, pooled_outputs)
			self.h_pool = tf.reshape(self.h_pool, [-1, int(length), num_filters_total])
		
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		with tf.name_scope("highway"):
			self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.probabilities = tf.nn.softmax(self.scores, name="probabilities")
			# self.predictions = tf.argmax(self.scores, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

