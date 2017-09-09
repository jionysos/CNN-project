import numpy as np
import tensorflow as tf
import random

data = np.loadtxt('/root//data//EleRhi//image_data_eleph_rhino.csv', delimiter=',')
# data = np.loadtxt('/root//jiwon//cc2.csv', delimiter=',')


print(type(data))
np.random.shuffle(data)

# data = data[:18176]
print(data.shape)

tf.set_random_seed(1234)

train_num = int(int(data.shape[0]) * .7)
print(train_num)  # 700
# print(data.shape)   # (1000, 5626)

x_train, t_train = data[:train_num, :-1], tf.one_hot(data[:train_num, -1], 2)
x_test, t_test = data[train_num:, :-1], tf.one_hot(data[train_num:, -1], 2)
# print(x_test.shape, t_test.shape)  # (300, 5625) (300, 2)

lr = 0.001
training_epochs = 70
batch_size = 256



class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def BN(self, inputs, training, scale=True, decay=0.99):
        self.inputs = inputs
        self.training = training
        return tf.layers.batch_normalization(inputs=self.inputs, training=self.training, scale=scale, momentum=decay)

    def _build_net(self):
        #         tf.reset_default_graph()
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, 5625])
        X_img = tf.reshape(self.X, [-1, 75, 75, 1])
        self.Y = tf.placeholder(tf.float32, [None, 2])

        #### CONV 1 ####
        W1 = tf.get_variable('W1', shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
        #         W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        #         L1 = self.BN(L1, self.training)
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
        print(L1.shape)  # (?, 38, 38, 32)
        # L1 = tf.reshape( L1, [-1, 32*38*38] )

        #### CONV 2
        W2 = tf.get_variable('W2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        #         W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        # 여기서 NORM
        #         L2 = self.BN(L2, self.training)
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
        print(L2.shape)

        #### CONV 3
        W3 = tf.get_variable('W3', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
        #         W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        # 여기서 NORM
        # L2 = self.batch_norm(L2, )
        #         L3 = self.BN(L3, self.training)
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
        print(L3.shape)

        #         L3 = tf.reshape( L3, [-1, 128*10*10] )

        #### CONV 4
        W7 = tf.get_variable('W7', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        #         W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        L7 = tf.nn.conv2d(L3, W7, strides=[1, 1, 1, 1], padding='SAME')
        # 여기서 NORM
        L7 = self.BN(L7, self.training)
        L7 = tf.nn.relu(L7)
        L7 = tf.nn.max_pool(L7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #         L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
        print(L7.shape)
        L7 = tf.reshape(L7, [-1, 256 * 5 * 5])

        #### AFF1
        W4 = tf.get_variable('W4', shape=[256 * 5 * 5, 512], initializer=tf.contrib.layers.xavier_initializer())
        #         W4 = tf.Variable(tf.random_normal([128*10*10, 500],stddev = 0.01))
        b4 = tf.Variable(tf.random_normal([512]))
        L4 = tf.matmul(L7, W4) + b4
        #         L4 = self.BN(L4, self.training)
        L4 = tf.nn.relu(L4)
        L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

        ### AFF2
        #         W5 = tf.Variable(tf.random_normal([625, 2], stddev = 0.01))
        W5 = tf.get_variable('W5', shape=[512, 1024], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([1024]))
        L5 = tf.matmul(L4, W5) + b5
        L5 = self.BN(L5, self.training)
        L5 = tf.nn.relu(L5)
        L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

        ### AFF3
        #       W5 = tf.Variable(tf.random_normal([625, 2], stddev = 0.01))
        W6 = tf.get_variable('W6', shape=[1024, 2], initializer=tf.contrib.layers.xavier_initializer())
        b6 = tf.Variable(tf.random_normal([2]))

        self.logits = tf.matmul(L5, W6) + b6
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        correct_predict = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    def get_acc(self, x_test, y_test, keep_prob=1., training=False):
        return self.sess.run(self.acc, feed_dict={self.X: x_test, self.Y: y_test.eval(session=self.sess),
                                                  self.keep_prob: keep_prob, self.training: training
                                                  })

    def train(self, x_data, y_data, keep_prob=.5, training=True):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data.eval(session=self.sess),
                                        self.keep_prob: keep_prob, self.training: training})


sess = tf.Session()
m1 = Model(sess, 'm1')

sess.run(tf.global_variables_initializer())



print('learning start')
for epoch in range(training_epochs):
    avg_cost = 0
    acc = 0
    total_size = int(x_train.shape[0])  # 700
    # print(total_size)
    avg_acc = 0
    total_batch = total_size / batch_size  # 7
    # print(total_batch)

    for i in range(0, total_size, batch_size):
        xt, yt = x_train[i:i + batch_size], t_train[i:i + batch_size]
        c, _ = m1.train(xt, yt)
        acc = m1.get_acc(xt, yt)
        avg_cost += c / total_batch
        avg_acc += acc / total_batch
    print('epoch: ', "%04d" % (epoch + 1), "cost: ", '{:.4f}'.format(avg_cost), '/   acc: ', '{:.4f}'.format(avg_acc))

# In[11]:


print('learning finished')

print('test')
total_test = int(x_test.shape[0])
test_num = total_test / batch_size
# print(test_num)
acc = 0
total = 0
for i in range(0, total_test, batch_size):
    acc = 0
    xt, tt = x_test[i:i + batch_size], t_test[i:i + batch_size]
    acc = m1.get_acc(xt, tt)
    # print(acc)
    total += acc

print('acc: ', total / test_num)
