'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
# import ipdb
import tensorflow as tf

from GlobalConstont import *

# from ln_lstm import LayerNormalizedLSTMCell
# from bnlstm import BNLSTMCell

from tensorflow.contrib.tensorboard.plugins import projector

class Model(object):
    def __init__(self, n_hidden, batch_size, p_keep_ff, p_keep_rc):
        '''n_hidden: number of hidden states
           p_keep_ff: forward keep probability
           p_keep_rc: recurrent keep probability'''
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        # if training:
        #     self.p_keep_ff = 1 - P_DROPOUT_FF
        #     self.p_keep_rc = 1 - P_DROPOUT_RC
        # else:
        #     self.p_keep_ff = 1
        #     self.p_keep_rc = 1
        self.p_keep_ff = p_keep_ff
        self.p_keep_rc = p_keep_rc
        # biases and weights for the last layer
        self.weights = {
            'out': tf.Variable(
                tf.random_normal([2 * n_hidden, EMBBEDDING_D * NEFF]), name='fc_layer_w')
        }
        self.biases = {
            'out': tf.Variable(
                tf.random_normal([EMBBEDDING_D * NEFF], name='fc_layer_bias'))
        }

        self.config = projector.ProjectorConfig()

    def return_config(self):
        return self.config

    def inference(self, x):
        '''The structure of the network'''
        # ipdb.set_trace()
        # four layer of LSTM cell blocks
        with tf.variable_scope('BLSTM1', initializer=tf.glorot_uniform_initializer()) as scope:
            # lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            z = lstm_fw_cell.get_weights()
            print(z)
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, x,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size, dtype=tf.float32)
            state_concate = tf.concat(outputs, 2)
        with tf.variable_scope('BLSTM2', initializer=tf.glorot_uniform_initializer()) as scope:
            # lstm_fw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell2, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell2, lstm_bw_cell2, state_concate,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate2 = tf.concat(outputs2, 2)
        with tf.variable_scope('BLSTM3', initializer=tf.glorot_uniform_initializer()) as scope:
            lstm_fw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell3, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell3, lstm_bw_cell3, state_concate2,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate3 = tf.concat(outputs3, 2)
        with tf.variable_scope('BLSTM4', initializer=tf.glorot_uniform_initializer()) as scope:
            lstm_fw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell4, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs4, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell4, lstm_bw_cell4, state_concate3,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate4 = tf.concat(outputs4, 2)
            
        with tf.variable_scope('BLSTM5', initializer=tf.glorot_uniform_initializer()) as scope:
            lstm_fw_cell5 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell5 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell5, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell5 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=True,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell5 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell5, input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs5, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell5, lstm_bw_cell5, state_concate4,
                sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                dtype=tf.float32)
            state_concate5 = tf.concat(outputs5, 2)
        # one layer of embedding output with tanh activation function
        with tf.variable_scope('FC_Layer') as scope:
            # out_concate = tf.reshape(state_concate4, [-1, self.n_hidden * 2])
            out_concate = tf.reshape(state_concate5, [-1, self.n_hidden * 2])

            emb_out = tf.matmul(out_concate,
                                self.weights['out']) + self.biases['out']
            emb_out = tf.nn.tanh(emb_out)
            reshaped_emb = tf.reshape(emb_out, [-1, NEFF, EMBBEDDING_D])
            # normalization before output
            normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2, name='Embedding')
            tf.summary.histogram('state_concate1', state_concate)
            tf.summary.histogram('state_concate2', state_concate2)
            # tf.summary.histogram('State_concate3', state_concate3)
            # tf.summary.histogram('State_conate4', state_concate4)
            tf.summary.histogram('Embedding', normalized_emb)
            embedding = self.config.embeddings.add()
            embedding.tensor_name = normalized_emb.name
            # Link this tensor to its metadata file (e.g. labels).
            # embedding.metadata_path = metadata
            # Saves a config file that TensorBoard will read during startup.

            return normalized_emb

    def loss(self, embeddings, Y, VAD):
        '''Defining the loss function'''
        with tf.name_scope('Loss_Computation'):
            embeddings_rs = tf.reshape(embeddings, shape=[-1, EMBBEDDING_D])          # shape = [200*129, 40]   // batch_size = 2, frames_per_sample = 100, NEFF = 129
            VAD_rs = tf.reshape(VAD, shape=[-1])                                      # shape = [200*129]
            # get the embeddings with active VAD
            embeddings_rsv = tf.transpose(
                tf.multiply(tf.transpose(embeddings_rs), VAD_rs))                     # shape = [200*129, 40]
            embeddings_v = tf.reshape(
                embeddings_rsv, [-1, FRAMES_PER_SAMPLE * NEFF, EMBBEDDING_D])         # shape = [2, 100*129, 40]
            # get the Y(speaker indicator function) with active VAD
            Y_rs = tf.reshape(Y, shape=[-1, 2])                                       # shape = [200*129, 2]
            Y_rsv = tf.transpose(
                tf.multiply(tf.transpose(Y_rs), VAD_rs))                              # shape = [200*129, 2]
            Y_v = tf.reshape(Y_rsv, shape=[-1, FRAMES_PER_SAMPLE * NEFF, 2])          # shape = [2, 100*129, 2]
            # fast computation format of the embedding loss function
            loss_batch = tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    embeddings_v, [0, 2, 1]), embeddings_v)) - \
                2 * tf.nn.l2_loss(
                    tf.matmul(tf.transpose(
                        embeddings_v, [0, 2, 1]), Y_v)) + \
                tf.nn.l2_loss(
                    tf.matmul(tf.transpose(
                        Y_v, [0, 2, 1]), Y_v))
            # v_vT = tf.matmul(embeddings, tf.transpose(embeddings, [0, 2, 1]))          # shape = [2, 100*129, 100*129]
            # V = tf.reshape(embeddings, [-1, FRAMES_PER_SAMPLE*NEFF, EMBBEDDING_D])
            # Y = tf.reshape(Y, [-1, FRAMES_PER_SAMPLE*NEFF, 2])
            # VT = tf.transpose(V, [0, 2, 1], name='V_transpose')
            # YT = tf.transpose(Y, [0, 2, 1], name='Y_transpose')
            # y_yT = tf.matmul(Y, YT, name='YYT')                            # shape = [2, 100*129, 100*129]
            
            # D = tf.multiply(y_yT, tf.eye(num_columns=FRAMES_PER_SAMPLE*NEFF, num_rows=FRAMES_PER_SAMPLE*NEFF, batch_shape=[self.batch_size]))
            # D = tf.sqrt(D, name='D_T')
            # with tf.variable_scope('Equation_3_See_Paper'):
            #     loss_batch = tf.nn.l2_loss(tf.matmul(VT, tf.matmul(D, V))) - 2*tf.nn.l2_loss(tf.matmul(VT, tf.matmul(D, Y))) + tf.nn.l2_loss(tf.matmul(YT, tf.matmul(D, Y)))
            loss_v = (loss_batch) / self.batch_size
            tf.summary.scalar('loss', loss_v)
            return loss_v

    def train(self, loss, lr):
        '''Optimizer'''
        with tf.name_scope('Optimization_Step'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
            # optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 200)
            train_op = optimizer.apply_gradients(
                zip(gradients, v))
            for index, grad in enumerate(gradients):
                tf.summary.histogram("{}-grad".format(gradients[index][1].name), gradients[index])
            for index, grad in enumerate(v):
                tf.summary.histogram("{}-v".format(v[index][1].name), v[index])            
            
            return train_op
