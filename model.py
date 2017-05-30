import random

import itertools
import numpy as np

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn

from mytensorflow import get_initializer
from rnn import get_last_relevant_rnn_output, get_sequence_length
from nn import multi_conv1d, highway_network

from read_data import DataSet



def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                tf.get_variable_scope().reuse_variables()
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)
    return models

class Model(object):

    # this index definition should be consistent with its definition of index in read_data.py's read_data() function.

    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        # Define forward inputs here
        N, JX, VW, VC, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, config.max_word_size


        self.vocab_size = config.total_word_vocab_size
        print("vocab size:", config.total_word_vocab_size)

        # Encoder input
        self.x = tf.placeholder('int32', [N, None], name='x')
        self.cx = tf.placeholder('int32', [N, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None], name='x_mask')
        self.x_length = tf.placeholder('int32', [N], name='x_length')

        # Decoder target
        self.y = tf.placeholder('int32', [N, None], name='y')
        self.cy = tf.placeholder('int32', [N, None, W], name='cy')
        self.y_mask = tf.placeholder('bool', [N, None], name='y_mask')
        self.y_length = tf.placeholder('int32', [N], name='y_length')

        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')        

        # Define misc
        self.tensor_dict = {}
        self.h_dim = config.hidden_size

        # Forward outputs / loss inputs
        self.logits = None
        self.var_list = None
        self.na_prob = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        N, JX, VW, VC, d, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, \
            config.hidden_size, config.max_word_size        
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size        



        # Getting word vector. For now, we only care about word_emb. Forget about char_emb.         
        with tf.variable_scope("emb"):
            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, JX, d]
                    Ay = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, JX, d]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['y'] = Ay
                    xx = Ax
                    yy = Ay

        # xx is the preocessed encoder input, 
        # yy should be derived from xx. 


        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                yy = highway_network(yy, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)



        self.tensor_dict['xx'] = xx
        self.tensor_dict['yy'] = yy

        self.encoder_inputs_embedded = xx

        self.decoder_train_inputs_embedded = yy
        self.decoder_train_length = self.y_length
        self.decoder_train_targets = self.x
        print("train_target:", self.decoder_train_targets)
  
        with tf.variable_scope("Encoder") as scope:
            encoder_cell = BasicLSTMCell(self.h_dim, state_is_tuple=True)      
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=encoder_cell,
                                inputs=self.encoder_inputs_embedded,
                                sequence_length=self.x_length,
                                dtype=tf.float32)
                )


        with tf.variable_scope("Decoder") as scope:

            decoder_cell = BasicLSTMCell(self.h_dim, state_is_tuple=True)

            print("self.decoder_train_inputs_embedded:", self.decoder_train_inputs_embedded)
            print("self.decoder_train_length:", self.decoder_train_length)
            helper = seq2seq.TrainingHelper(self.decoder_train_inputs_embedded, self.decoder_train_length)
            # Try schduled training helper. It may increase performance. 

            decoder = seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=self.encoder_state
            )
            # Try AttentionDecoder.
            self.decoder_outputs_train, self.decoder_state_train = seq2seq.dynamic_decode(
                    decoder, scope=scope,
                )

            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
            print("shape of self.decoder_outputs_train.rnn_output", self.decoder_outputs_train.rnn_output)
            self.decoder_logits_train = output_fn(self.decoder_outputs_train.rnn_output)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')
        
    def _build_loss(self):
        config = self.config
        
        # Try tf.nn.sampled_softmax_loss if the softmax is too large.         
        self.loss = seq2seq.sequence_loss(
            logits=self.decoder_logits_train,
            targets=self.decoder_train_targets,
            weights=tf.cast(self.x_mask, tf.float32),
            name='loss'
            )
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            print('opname:', ema_var.op.name)
            print('var:', ema_var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, JX, VW, VC, d, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        # Assume that we are using auto-encoder style, encoder decoder. input is the same as the output.

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for sent in batch.data['x_list']) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for sent in batch.data['x_list'])

            JX = min(JX, max(new_JX))
        # +1 for start of sentence "EOS" symbol
        x = np.zeros([N, JX+1], dtype='int32')
        cx = np.zeros([N, JX+1, W], dtype='int32')        
        x_mask = np.zeros([N, JX+1], dtype='bool')
        x_length = np.zeros([N], dtype='int32')


        # +2 for start of sentence "GO" and "EOS" symbol
        y = np.zeros([N, JX+2], dtype='int32')
        cy = np.zeros([N, JX+2, W], dtype='int32')
        y_mask = np.zeros([N, JX+2], dtype='bool')
        y_length = np.zeros([N], dtype='int32')
        
        
        feed_dict[self.x] = x
        feed_dict[self.cx] = cx
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.x_length] = x_length
        
        
        feed_dict[self.y] = y
        feed_dict[self.cy] = cy
        feed_dict[self.y_mask] = y_mask
        feed_dict[self.y_length] = y_length




        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x_list']
        CX = batch.data['cx_list']

        def _get_word(word):
            """
            return index of the word from the preprocessed dictionary. 
            """
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1


        
        # replace char data to index. 
        EOS = _get_word("-EOS-")
        PAD = _get_word("-NULL-")
        for i, xi in enumerate(X):
            for j, xij in enumerate(xi):
                if j == config.max_sent_size:
                    break
                each = _get_word(xij)
                assert isinstance(each, int), each
                x[i, j] = each
                x_mask[i, j] = True
            x[i, len(xi)] = EOS
            x_length[i] = len(xi)+1

            y[i] = np.concatenate(([_get_word("-GO-")], x[i]))
            for idx in range(x_length[i]+1, JX+2):
                y[i, idx] = PAD
            y_length[i] = JX+1
            y_mask = np.concatenate(([True], x_mask[i]))


        for i, cxi in enumerate(CX):
            for j, cxij in enumerate(cxi):
                if j == config.max_sent_size:
                    break           
                for k, cxijk in enumerate(cxij):
                    if k == config.max_word_size:
                        break
                    cx[i, j, k] = _get_char(cxijk)
          

        return feed_dict
