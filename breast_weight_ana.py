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
    result_dict = dict()
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
                        # print("variable name: ", key)
                        # print("weight_shape: ", np.shape(weight_value))
                        print(weight_value)
                        count = count + 1
                        result_dict[key] = weight_value


            else:
                print('No checkpoint file found')
    # count和breast的shape中的column 列数一致，breast中包含label
    # count中有针对bias_weight的统计，都等于len(feature_columns)+1
    print("count: ", count)

    return result_dict


def is_has_big_weight(result):
    """
    判断是否有大于1的权重
    :return:
    """
    bigger_than_one = dict()
    for k in result.keys():
        weights = result[k]
        for weight in weights:
            if isinstance(weight, list) or isinstance(weight, np.ndarray):
                for wi in weight:
                    if wi > 1:
                        bigger_than_one[k] = weights
            elif weight > 1:
                bigger_than_one[k] = weights

    # 输出大于1的权重
    for k in bigger_than_one.keys():
        v = bigger_than_one[k]
        print("k: ", k, ", v: ", v)


if __name__ == '__main__':
    result = two_example()
    is_has_big_weight(result)
