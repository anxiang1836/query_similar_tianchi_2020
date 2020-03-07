from bert4keras.bert import build_bert_model
from tensorflow.keras.layers import Dropout, Dense, subtract, multiply, maximum, Concatenate
from tensorflow.keras import Model, Input
import tensorflow as tf
from typing import List


class SiameseBertModel:
    def __init__(self, config_path: str, checkpoint_path: str, dense_units: List[int]):
        """
        初始化预训练的模型参数
        :param config_path    :
        :param checkpoint_path:
        :param dense_units: List[int] 从Conv层到Output的各Dense层的神经元个数
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dense_units = dense_units

    def get_model(self):
        # 加载预训练模型
        bert = build_bert_model(
            config_path=self.config_path, checkpoint_path=self.checkpoint_path,
            with_pool=True, return_keras_model=False, model="albert")

        # query1 = bert.model()([q1_token_in, q1_seg_in])
        # query2 = bert.model()([q2_token_in, q2_seg_in])

        q1_x_in = Input(shape=(None, ), name='Input-Token-q1')
        q2_x_in = Input(shape=(None,), name='Input-Token-q2')
        q1_s_in = Input(shape=(None,), name='Input-Segment-q1')
        q2_s_in = Input(shape=(None,), name='Input-Segment-q2')



        input_layer = bert.model.input
        input_layer.extend(bert.model.input)

        query1 = Dropout(rate=0.1)(bert.model.output)
        query2 = Dropout(rate=0.1)(bert.model.output)

        # |q1-q2| 两特征之差的绝对值
        sub = tf.abs(subtract([query1, query2]))
        # q1*q2 两特征按元素相乘
        mul = multiply([query1, query2])
        # max(q1,q2)^2 两特征取最大元素的平方
        max_square = multiply([maximum([query1, query2]), maximum([query1, query2])])

        merge_layers = Concatenate()([query1, query2, sub, mul, max_square])

        merge_layers = Dropout(rate=0.1)(merge_layers)

        fc = None
        for i in range(len(self.dense_units)):
            if i == 0:
                fc = Dense(self.dense_units[i], activation="relu", kernel_initializer=bert.initializer)(merge_layers)
            elif i == len(self.dense_units) - 1:
                fc = Dense(units=2, activation='softmax', kernel_initializer=bert.initializer)(fc)
            else:
                fc = Dense(self.dense_units[i], activation="relu", kernel_initializer=bert.initializer)(fc)

        model = Model(input_layer, fc)
        model.summary()
        return model
