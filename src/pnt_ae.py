import os, time
import os.path as osp
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar
from tflearn import is_training
from structural_losses.tf_nndistance import nn_distance
from structural_losses.tf_approxmatch import approx_match, match_cost
from general_utils import rand_rotation_matrix

blue = lambda x:'\033[94m' + x + '\033[0m'

'''
    配置类
'''
class Configuration():
    def __init__(self, n_input, training, encoder, decoder, encoder_args={}, decoder_args={},
                 training_epochs=200, batch_size=10, learning_rate=0.001, model_path=None,
                 train_dir=None, loss='chamfer', n_output=None):

        # Parameters for any AE
        self.n_input = n_input
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        self.training = training
        self.model_path = model_path

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_dir = train_dir
        self.training_epochs = training_epochs

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        res = ''
        for key, val in self.__dict__.items():
            if callable(val):
                v = val.__name__
            else:
                v = str(val)
            res += '%20s: %s\n' % (str(key), str(val))
        return res

'''
    PointNetAutoEncoder 实现代码
'''
class PointNetAutoEncoder():
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, name, configuration):
        c = configuration
        self.configuration = configuration
        self.graph = tf.get_default_graph()
        self.name = name

        with tf.variable_scope(self.name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)
            # 输入
            in_shape = [None] + c.n_input
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = self.x

            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            # 输出
            self.x_reconstr = tf.reshape(layer, [-1, c.n_output[0], c.n_output[1]])
            # loss
            self._create_loss()
            # BP
            if c.training:
                self._setup_optimizer()

            # Launch the session
            config = tf.ConfigProto()
            self.sess = tf.Session(config=config)

            if c.training:
                is_training(True, session=self.sess)
                # Summaries
                self.merged_summaries = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter(osp.join(c.train_dir, 'summaries'), self.graph)
                # Saver
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                if c.model_path is not None:
                    self.restore_model(c.model_path)
                else:
                    # Initializing the variables
                    self.init = tf.global_variables_initializer()
                    self.sess.run(self.init)
            else:
                is_training(False, session=self.sess)
                self.saver = tf.train.Saver()
                self.restore_model(c.model_path)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration

        self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, 50,
                    decay_rate=0.98, staircase=True, name="learning_rate_decay")
        self.lr = tf.maximum(self.lr, 1e-5)
        tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, only_fw=False):
        c = self.configuration
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = c.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        bar = ProgressBar(maxval=n_batches).start()
        for batch_idx in range(n_batches):
            batch_i, _ = train_data.next_batch(batch_size)
            batch_i = batch_i.dot(rand_rotation_matrix())

            # train
            _, loss, _ = self.sess.run(fetches=(self.train_step, self.loss, self.x_reconstr),
                                       feed_dict={self.x: batch_i})

            # Compute average loss
            epoch_loss += loss
            bar.update(batch_idx)
        bar.finish()
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def train(self, train_data, log_file=None):
        c = self.configuration
        assert c.training

        stats = []
        if not osp.exists(c.train_dir):
            os.makedirs(c.train_dir)

        for _ in range(c.training_epochs):
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))

            print(blue('----- epoch %d -----' % epoch))
            loss, duration = self._single_epoch_train(train_data, c)
            train_data.shuffle_data()
            print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
            log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if epoch % 1 == 0:
                checkpoint_path = osp.join(c.train_dir, 'models.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

    def restore_model(self, model_path):
        '''
            load existing model for inferencing OR continue training
        '''
        self.saver.restore(self.sess, model_path)
        print(blue('restore sess at epoch %d' % self.epoch.eval(session=self.sess)))

    def _eval_one(self, feed_data):
        '''
            evalutate one data
        '''
        feed_data = np.expand_dims(feed_data, axis=0)
        loss, reconstr = self.sess.run(fetches=(self.loss, self.x_reconstr),
                                       feed_dict={self.x: feed_data})
        reconstr = np.reshape(reconstr, newshape=[-1,3])
        return loss, reconstr

