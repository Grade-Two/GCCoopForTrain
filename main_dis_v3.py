import tensorflow as tf
import numpy as np


tf.flags.DEFINE_float("learning_rate", 0.00003, "Initial learning rate")
tf.flags.DEFINE_integer("steps_to_validate", 100, "tps of validation")
tf.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps")
tf.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker")
tf.flags.DEFINE_string("job_name", "ps", "job_name")
tf.flags.DEFINE_integer("task_index", 0, "Index")
FLAGS = tf.flags.FLAGS
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
        global_step = tf.train.get_or_create_global_step()
        input = tf.placeholder("float")
        label = tf.placeholder("float")
        weight = tf.get_variable("weights", [1], tf.float32, initializer=tf.random_normal_initializer())
        bias = tf.get_variable("bias", [1], tf.float32, initializer=tf.random_normal_initializer())
        pred = tf.multiply(input, weight) + bias
        loss_value = tf.reduce_mean(tf.square(label - pred))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.summary.scalar("cost", loss_value)
        summary_op = tf.summary.merge_all()
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir="./check_point/", init_op=init_op,
                                 saver=saver, global_step=global_step, save_model_secs=60)
        with sv.managed_session(server.target)as sess:
            step = 0
            while step < 1000000:
                train_x = np.random.randn(1)
                train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                _, loss_v, step = sess.run([train_op, loss_value, global_step],
                                           feed_dict={input: train_x, label: train_y})
                if step % steps_to_validate == 0:
                    w, b = sess.run([weight, bias])
                    print("step:%d,weight %f,bais:%f,loss:%f" % (step, w, b, loss_v))
                    
                    
if __name__ == "__main__":
    tf.app.run()
