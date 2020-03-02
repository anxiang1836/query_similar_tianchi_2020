import abc
import numpy as np
from typing import List
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense


class BasicModel:
    def __init__(self, emb_matrix: np.ndarray, word2idx: np.ndarray, dense_units: List[int],
                 label_count: int, category_count: int, query_len: int, shared: bool = True):
        """
        :param emb_matrix     : embedding矩阵
        :param word2idx       : word转idx的矩阵
        :param dense_units    : 到Output的各Dense层的神经元个数
        :param label_count    : int 待预测的标签数
        :param category_count : int 原始数据中query类型数量
        :param query_len      : int 原始数据中query的最大长度
        :param shared         : bool 双塔是否共享特征提取的CNN权重
        """
        self.emb_matrix = emb_matrix
        self.word2idx = word2idx
        self.dense_units = dense_units
        self.label_count = label_count
        self.category_count = category_count
        self.query_len = query_len
        self.shared = shared

    @abc.abstractmethod
    def build_feature(self) -> Sequential:
        # 用于构建特征的抽象方法，子类必须实现该方法
        pass

    def get_model(self):
        """
        用于构建完整模型，即build_feature + Dense
        :return:
            model
        """
        input_category = Input(shape=(self.category_count,))
        input_query1 = Input(shape=(self.query_len,))
        input_query2 = Input(shape=(self.query_len,))

        # Layer1: 特征抽取层
        if self.shared:
            # 调用了1次Model，是双塔共享模式
            model = self.build_feature()

            query_1 = model(input_query1)
            query_2 = model(input_query2)
            merge_layers = Concatenate()([query_1, query_2, input_category])

        else:
            # 调用了2次Model,是双塔非共享模型
            query_1 = self.build_feature()(input_query1)
            query_2 = self.build_feature()(input_query2)
            merge_layers = Concatenate()([query_1, query_2, input_category])

        # Layer2：全连接层
        fc = None
        for i in range(len(self.dense_units)):
            if i == 0:
                fc = Dense(self.dense_units[i], activation="relu")(merge_layers)
            elif i == len(self.dense_units) - 1:
                fc = Dense(1, activation='sigmoid')(fc)
            else:
                fc = Dense(self.dense_units[i], activation="relu")(fc)

        model = Model(inputs=[input_category, input_query1, input_query2], outputs=[fc])
        model.summary()

        return model
