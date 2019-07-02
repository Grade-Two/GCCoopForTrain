import tensorflow as tf
import numpy as np
import pickle

label_bytes = 1
image_size = 32
image_depth = 3
image_bytes = image_size * image_size * image_depth
num_classes = 10


class Reader:
    def __init__(self, filenames):
        self.filenames = filenames
        self.label_bytes = 1
        self.image_size = 32
        self.image_depth = 3
        self.image_bytes = image_size * image_size * image_depth
        self.num_classes = 10
        self.buffer_images, self.buffer_labels = self._read_all()
        self.pos = 0
        self.epoch = 0

    def _read_all(self):
        buffer_images = []
        buffer_labels = []
        for filename in self.filenames:
            f = open(filename, 'rb')
            data = pickle.load(f, encoding='bytes')
            labels = data[b'labels']
            for label in labels:
                one_hot = [0] * self.num_classes
                one_hot[label] = 1
                buffer_labels.append(one_hot)
            images = np.reshape(data[b'data'], (-1, 3, 32, 32))
            images = images.transpose((0, 2, 3, 1))
            buffer_images.extend(images.tolist())
            f.close()
        return buffer_images, buffer_labels

    def next_batch(self, batch_size):
        if self.pos + batch_size <= len(self.buffer_labels):
            images = self.buffer_images[self.pos: self.pos+batch_size]
            labels = self.buffer_labels[self.pos: self.pos+batch_size]
            self.pos += batch_size
            if self.pos >= len(self.buffer_labels):
                self.pos = self.pos - len(self.buffer_labels)
                self.epoch += 1
            return images, labels
        else:
            images = self.buffer_images[self.pos:]
            labels = self.buffer_labels[self.pos:]
            last = batch_size - len(labels)
            images.extend(self.buffer_images[:last])
            labels.extend(self.buffer_labels[:last])
            self.pos = last
            self.epoch += 1
            return images, labels

    def clear(self):
        self.pos = 0
        self.epoch = 0


def read_cifar10(data_file, batch_size):
    record_bytes = label_bytes + image_bytes
    data_files = tf.gfile.Glob(data_file)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]), [image_depth, image_size, image_size])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    example_queue = tf.RandomShuffleQueue(capacity=16*batch_size, min_after_dequeue=8*batch_size,
                                          dtypes=[tf.float32, tf.int32],
                                          shapes=[[image_size, image_size, image_depth], [1]])
    num_threads = 16
    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    # labels = tf.sparse_to_dense(tf.concat(values=[0, labels], axis=1), [batch_size, num_classes], 1.0, 0.0)

    return images, labels

'''
if __name__ == '__main__':
    reader = Reader(['cifar-10-python\\cifar-10-batches-py\\data_batch_1'])
    print(reader.next_batch(2))
    print(reader.next_batch(4))
'''
