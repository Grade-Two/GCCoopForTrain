import tensorflow as tf
import time
import math
import numpy as np

import Cifar10Reader
import ResNet_Model

flags = tf.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('train_dir', 'tmp/train', '')
flags.DEFINE_string('data_dir', 'cifar-10-python/cifar-10-batches-py/', 'Directory  for storing mnist data')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '192.168.32.145:22221', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '192.168.32.146:22221, 192.168.32.160:22221',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("sync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
# 选择计算设备
flags.DEFINE_string('device', '/gpu:0', 'format like "/cpu:1" or "/gpu:2"')

FLAGS = flags.FLAGS


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
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    train_reader = Cifar10Reader.Reader(['cifar-10-python\\cifar-10-batches-py\\data_batch_1',
                                         'cifar-10-python\\cifar-10-batches-py\\data_batch_2',
                                         'cifar-10-python\\cifar-10-batches-py\\data_batch_3',
                                         'cifar-10-python\\cifar-10-batches-py\\data_batch_4',
                                         'cifar-10-python\\cifar-10-batches-py\\data_batch_5'])
    if is_chief is True:
        test_reader = Cifar10Reader.Reader(['cifar-10-python\\cifar-10-batches-py\\test_batch'])
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        # step
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
        # input
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None, 10])
        # train
        logits, _ = ResNet_Model.resnet_v2_50(x, 10)
        train_op, loss_op = train(logits, y)
        # test
        test_op = test(logits, y)
        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.train_dir, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)
        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initialized...' % FLAGS.task_index)
        sess = sv.prepare_or_wait_for_session(server.target)
        print('Worker %d: Session initialization  complete.' % FLAGS.task_index)

        local_step = 0
        start_time = time.time()
        while True:
            train_images, train_labels = train_reader.next_batch(FLAGS.batch_size)
            train_images = tf.cast(train_images, tf.float32)
            train_labels = tf.cast(tf.one_hot(train_labels, 10), tf.int32)
            train_images, train_labels = sess.run([train_images, train_labels])
            _, loss, step = sess.run([train_op, loss_op, global_step], feed_dict={x: train_images, y: train_labels})
            local_step += 1
            if local_step % 100 == 0:
                duration = time.time() - start_time()
                print('Worker %d: training step %d dome (global step:%d)' % (FLAGS.task_index, local_step, step))
                if is_chief is True:
                    correct_count = 0.0
                    input_count = 0
                    while test_reader.epoch < 1:
                        test_images, test_labels = test_reader.next_batch(FLAGS.batch_size)
                        test_images = tf.cast(test_images, tf.float32)
                        test_labels = tf.cast(tf.one_hot(test_labels, 10), tf.int32)
                        test_images, test_labels = sess.run([test_images, test_labels])
                        input_count += len(test_labels)
                        correct_count += sess.run(test_op, feed_dict={x: test_images, y: test_labels})
                    print('time: %.5f, loss: %.3f, acc: %.3f' % (duration, loss, correct_count/input_count))
                    test_reader.clear()
                    start_time = time.time()
            if step >= FLAGS.train_steps:
                break
        sess.close()

if __name__ == '__main__':
    main()
