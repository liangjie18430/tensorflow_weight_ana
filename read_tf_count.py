# encoding:utf8
from __future__ import print_function
import tensorflow as tf
import datetime

def name_to_features_func():
	seq_length = 384
	name_to_features = {
		"unique_ids": tf.FixedLenFeature([], tf.int64),
		"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

	name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
	name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
	# 返回example的定义格式
	return  name_to_features


def read_and_decode(input_file):
    d = tf.data.TFRecordDataset(input_file)
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # 接续数据
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example
    # 获取name_to_features
    name_to_features = name_to_features_func()
    # 使用name_to_features_func进行解析
    ds = d.map(lambda record:_decode_record(record,name_to_features))
    return ds

def count(ds):
    """
    ds:一个tensorflow的dataset的对象
    """
    
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.InteractiveSession()

    # 获取
    i = 0
    while True:
        # 不断的获得下一个样本
        try:
            # 获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂
            next_data = sess.run(next_element)
            i = i + 1
            if i < 10:
                print("i: {},data: {}".format(i,next_data))
        # 如果遍历完了数据集，则返回错误
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
    print('the all record count is {}.'.format(i))

if __name__=='__main__':
    start = datetime.datetime.now()
    ds = read_and_decode('train.tf_record')
    count(ds)
    end = datetime.datetime.now()
    print('the count all cost {}.'.format(end-start))
