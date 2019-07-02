import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist.input_data as input_data
import ResNet_Model
import Cifar10Reader
import time

tf.flags.DEFINE_string("ps_hosts", "c240g5-110201.wisc.cloudlab.us:2223", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "c240g5-110109.wisc.cloudlab.us:2224,c240g5-110207.wisc.cloudlab.us:2224", "worker hosts")
tf.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.flags.DEFINE_integer("num_workers", 1, "Number of workers")
tf.flags.DEFINE_boolean("is_sync", True, "using synchronous training or not")
tf.flags.DEFINE_boolean("data_dir", "data", "")
tf.flags.DEFINE_integer("batch_size", 32, "")
FLAGS = tf.flags.FLAGS


def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # create the cluster configured by `ps_hosts' and 'worker_hosts'
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # create a server for local task
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()  # ps hosts only join
    elif FLAGS.job_name == "worker":
        # workers perform the operation
        # ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(FLAGS.num_ps)

        # Note: tf.train.replica_device_setter automatically place the paramters (Variables)
        # on the ps hosts (default placement strategy:  round-robin over all ps hosts, and also
        # place multi copies of operations to each worker host
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            # load mnist dataset
            mnist = input_data.read_data_sets("mnist", one_hot=True)
            # the model
            images = tf.placeholder(tf.float32, [None, 32, 32, 3])
            labels = tf.placeholder(tf.int32, [None, 10])
            logits, _ = ResNet_Model.resnet_v2_50(images, 10)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=2000000)]
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)
            if FLAGS.is_sync:
                # asynchronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(worker_hosts),
                                                           total_num_replicas=len(worker_hosts))
                # create the hook which handles initialization and queues
                hooks.append(optimizer.make_session_run_hook((FLAGS.task_index == 0)))
            train_op = optimizer.minimize(loss, global_step=global_step,
                                          aggregation_method=tf.AggregationMethod.ADD_N)
            # eval
            correct_prediction = tf.equal(tf.argmax(labels, 1),
                                          tf.argmax(tf.reshape(logits, (FLAGS.batch_size, 10)), 1))
            test_op = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            train_reader = Cifar10Reader.Reader(['cifar-10-python\\cifar-10-batches-py\\data_batch_1',
                                                 'cifar-10-python\\cifar-10-batches-py\\data_batch_2',
                                                 'cifar-10-python\\cifar-10-batches-py\\data_batch_3',
                                                 'cifar-10-python\\cifar-10-batches-py\\data_batch_4',
                                                 'cifar-10-python\\cifar-10-batches-py\\data_batch_5'])
            if FLAGS.task_index == 0:
                test_reader = test_reader = Cifar10Reader.Reader(['cifar-10-python\\cifar-10-batches-py\\test_batch'])
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   hooks=hooks) as mon_sess:
                begin_time = time.time()    # total time
                start_time = time.time()    # 100 steps time
                test_time = 0.0             # test time
                while not mon_sess.should_stop():
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    img_batch, label_batch = train_reader.next_batch(FLAGS.batch_size)
                    _, ls, step = mon_sess.run([train_op, loss, global_step],
                                               feed_dict={images: img_batch, labels: label_batch})
                    if step % 100 == 0:
                        print("Train step %d, loss: %f, time: %.4f" % (step, ls, time.time() - start_time))
                        if step % 1000 == 0 and FLAGS.task_index == 0:
                            temp_time = time.time()
                            correct_count = 0.0
                            input_count = 0
                            test_reader.clear()
                            while test_reader.epoch < 1:
                                test_images, test_labels = test_reader.next_batch(FLAGS.batch_size)
                                input_count += len(test_labels)
                                correct_count += mon_sess.run(test_op, feed_dict={images: test_images, labels: test_labels})
                            test_time += time.time() - temp_time
                            print('time: %.5f, acc: %.3f' % (time.time() - begin_time, correct_count / input_count))
                        start_time = time.time()


if __name__ == "__main__":
    tf.app.run()