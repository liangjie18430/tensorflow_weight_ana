# -* coding:utf8 *-
from __future__ import print_function
import numpy as np

"""
用于分析权重，查看哪些权重不符合要求
"""

# 原文地址:https://blog.csdn.net/AManFromEarth/article/details/81057577
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def two_example():
    count = 0
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # br_logs是用breast的dataframe训练wide and deep所产生的文件
            ckpt = tf.train.get_checkpoint_state('./br_logs')
            if ckpt and ckpt.model_checkpoint_path:
                reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
                all_variables = reader.get_variable_to_shape_map()
                # w1 = reader.get_tensor("conv1/weight")

                for key in all_variables:
                    weight_value = reader.get_tensor(key)

                    if (key.endswith("weights")):
                        print("variable name: ", key)
                        print("weight_shape: ", np.shape(weight_value))
                        print(weight_value)
                        count = count + 1


            else:
                print('No checkpoint file found')

    print("count: ",count)
if __name__ == '__main__':
    two_example()
