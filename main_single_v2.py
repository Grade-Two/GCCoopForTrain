import tensorflow as tf
import time
import numpy as np
from tensorflow.contrib.slim import nets

import Cifar10Reader

slim = tf.contrib.slim


tf.flags.DEFINE_integer('num_clones', 1, '')
tf.flags.DEFINE_bool('clone_on_cpu', False, '')
tf.flags.DEFINE_integer('task', 0, 'worker or ps id')
tf.flags.DEFINE_string('device', '/gpu:0', 'format like "/cpu:1" or "/gpu:2"')
tf.flags.DEFINE_integer('batch_size', 128, 'Training batch size ')
tf.flags.DEFINE_integer('worker_replicas', 1, 'number of workers')
tf.flags.DEFINE_integer('ps_tasks', 1, 'number of ps')
tf.flags.DEFINE_string('dataset_dir', 'cifar-10-python\\cifar-10-batches-py\\', 'direction of dataset')
FLAGS = tf.flags.FLAGS

data_dir = FLAGS.dataset_dir


def train(logits, labels):
    with tf.device(FLAGS.device):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy_mean)
    return train_step, cross_entropy_mean


def test(logits, labels):
    with tf.device(FLAGS.device):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(tf.reshape(logits, (32, 10)), 1))
        count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return count


def main():
    batch_size = FLAGS.batch_size
    train_files = [FLAGS.dataset_dir+'data_batch_'+str(i) for i in range(1, 6)]
    test_files = [FLAGS.dataset_dir+'test_batch']
    print('train data: ', train_files)
    print('test data: ', test_files)
    train_reader = Cifar10Reader.Reader(train_files)
    test_reader = Cifar10Reader.Reader(test_files)
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None, 10])
        is_training = tf.placeholder(tf.bool, name='is_training')
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, _ = nets.resnet_v1.resnet_v1_50(x, num_classes=10, is_training=is_training)
        # 训练
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net, name='cross_entropy_per_example')
        loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy')
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss_op)
        # 测试
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf.reshape(net, (32, 10)), 1))
        test_op = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            start_time = time.time()
            while train_reader.epoch < 5:
                train_images, train_labels = train_reader.next_batch(batch_size)
                # print(train_images)
                train_images = tf.cast(train_images, tf.float32)
                train_labels = tf.cast(tf.one_hot(train_labels, 10), tf.int32)
                train_images, train_labels = sess.run([train_images, train_labels])
                _, loss = sess.run([train_op, loss_op], feed_dict={x: train_images, y: train_labels, is_training: True})
                step += 1
                print(step)
                if step % 100 == 0:
                    duration = time.time() - start_time
                    correct_count = 0.0
                    input_count = 0
                    while test_reader.epoch < 1:
                        test_images, test_labels = test_reader.next_batch(batch_size)
                        test_images = tf.cast(test_images, tf.float32)
                        test_labels = tf.cast(tf.one_hot(test_labels, 10), tf.int32)
                        test_images, test_labels = sess.run([test_images, test_labels])
                        input_count += len(test_labels)
                        correct_count += sess.run(test_op, feed_dict={x: test_images, y: test_labels, is_training: False})
                    print('step: %d: time: %.5f, loss: %.3f, acc: %.3f'
                          % (step, duration, loss, correct_count / input_count))
                    test_reader.clear()
                    start_time = time.time()

if __name__ == '__main__':
    main()
