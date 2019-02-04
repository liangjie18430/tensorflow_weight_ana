# -* coding:utf8 *-
from __future__ import print_function
import pandas as pd
from sklearn import datasets
from tensorflow.python.estimator.run_config import RunConfig
import shutil
from sklearn.model_selection import train_test_split
breast_cancer = datasets.load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
feature_names = [name.replace(' ', '') for name in breast_cancer.feature_names]
x_pd = pd.DataFrame(x, columns=feature_names)
y_pd = pd.DataFrame(y, columns=['label'])
all_pd = pd.concat([x_pd, y_pd], axis=1)

import time
import tensorflow as tf


# 根据不同的列构建不同的tf_column的输入
def construct_tf_column(df, continuous_result_columns, discrete_result_columns):
    # 如果是连续的列，肯定是实质性
    con_tf_columns = []
    for column in continuous_result_columns:
        tf_column = tf.feature_column.numeric_column(key=column)
        con_tf_columns.append(tf_column)
    cate_tf_columns = []

    # todo 针对某些连续的列，进行离散化的尝试，可以翻译已经做好的LR
    # 默认将所有离散后的列，作为基本的columns

    # 可以构造交叉特征，交叉特征必须是离散的列之间,交叉特征只能输入给线性部分,注意近地铁弄出来的拼音为jindetie,注意pypinyin解析出的错误

    # 对于使用vocabulary_list离散化的列,使用indicator传入,或者embedding传入
    cate_tf_id_columns = [tf.feature_column.indicator_column(tf_column) for tf_column in cate_tf_columns]

    # 对于使用hash_bucket的列，使用embedding_column传入，dimension中可能需要进行合理的设置

    # 对于deep部分,将数值型的和离散的列传入
    deep_columns = con_tf_columns
    # 对于wide部分，直接传入交叉特征,和其他的离散特征
    wide_columns = con_tf_columns
    return wide_columns, deep_columns, con_tf_columns, cate_tf_columns


def input_fn(df, batch_size=500, num_epochs=2, shuffle=True):
    data_df = df.drop(["label"], axis=1)
    label_df = df["label"]
    return tf.estimator.inputs.pandas_input_fn(x=data_df, y=label_df, num_epochs=num_epochs, shuffle=shuffle,
                                               batch_size=batch_size, num_threads=1)


tf.set_random_seed(12)
run_config = RunConfig(tf_random_seed=12)


# 构建模型,头肩模型时传入了相关的
def build_estimator(model_dir=None, model_type="wide_deep", wide_columns=None, deep_columns=None):
    """
    定义构建模型的方法
    :param model_dir:
    :param model_type:
    :return:
    """
    hidden_units = [100, 50]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    run_config = tf.estimator.RunConfig().replace(session_config=tf_config, tf_random_seed=12)
    if model_type == 'wide':
        # 使用线性模型
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        # 使用深度学习模型
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        # 使用wide and deep模型
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)






def get_base_columns(df):
    if hasattr(df, "columns"):
        df_columns = df.columns
    else:
        raise ValueError("df has no attr columns")
    continuous_result_columns = []
    discrete_result_columns = []
    for column in df_columns:
        continuous_result_columns.append(column)

    continuous_result_columns.remove("label")
    # 查看真正的哪些column没有被匹配上
    all_result_columns = set(continuous_result_columns).union(set(discrete_result_columns))
    other_columns = set(df_columns).difference(all_result_columns)
    print("the columns that not match:\n", other_columns)
    return continuous_result_columns, discrete_result_columns


def model_metric(model, df, message):
    """
    model为训练好的模型
    df为需要评估的dataframe
    message 为需要输出的额外信息

    """
    start_time = time.clock()
    results = model.evaluate(input_fn=input_fn(df, num_epochs=10, shuffle=False))
    # results = model.evaluate(input_fn=lambda :input_fn_file(drop_three_filename, num_epochs=1, shuffle=False,feature_columns=train_df.columns,default_value=default_value))
    end_time = time.clock()
    print(message, "eval time cost: %f s" % (end_time - start_time))
    # Display evaluation metrics,输出每次的评估结果
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    pass


train_df, test_df = train_test_split(all_pd, stratify=all_pd['label'], random_state=42, test_size=0.2)
# 获取离散的列和非离散的列
continuous_result_columns, discrete_result_columns = get_base_columns(train_df)

print(continuous_result_columns)

model_dir = "/Users/admin/workspace/git_workspace/recsys-algo/data/tfcord_process/test_tensor_board/br_logs"
shutil.rmtree(model_dir, ignore_errors=True)
wide_columns, deep_columns, con_tf_columns, cate_tf_columns = construct_tf_column(train_df, continuous_result_columns,
                                                                                  discrete_result_columns)
# model = build_estimator(wide_columns=wide_columns,deep_columns=deep_columns,model_type="deep")
model = build_estimator(wide_columns=wide_columns, deep_columns=deep_columns, model_type='wide',model_dir=model_dir)
# model = build_estimator(wide_columns=wide_columns,deep_columns=deep_columns,model_dir=model_dir)
model.train(input_fn=input_fn(train_df, num_epochs=15, shuffle=False))
model_metric(model, train_df, "df_train metric.")
model_metric(model, test_df, "df_test metric.")
