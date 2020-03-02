from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Embedding, Concatenate, LSTM, Bidirectional, Activation, Masking, \
    Dense
import numpy as np
from typing import List


class SimeseRnnModel(object):
    def __init__(self, emb_matrix: np.ndarray, word2idx: np.ndarray, hidden_units: List[int], dense_units: List[int],
                 label_count: int, category_count: int, query_len: int, mask_zero: bool = True,
                 bidirection: bool = True, shared: bool = True):
        """
        初始化模型参数

        :param emb_matrix    : embedding矩阵
        :param word2idx      : word转idx的矩阵
        :param hidden_units  : List[int] LSTM的hidden Unit个数
        :param dense_units   : List[int] 从LSTM层到Output的各Dense层的神经元个数
        :param label_count   : int 待预测的标签数
        :param category_count: int 原始数据中query类型数量
        :param query_len     : int 原始数据中query的最大长度
        :param mask_zero     : bool 是否在Embedding层对zero进行Mask处理
        :param bidirection   : bool 每层是否为双向rnn，默认为True
        :param shared        : bool 双塔是否共享特征提取的CNN权重
        """
        self.emb_matrix = emb_matrix
        self.word2idx = word2idx
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.label_count = label_count
        self.category_count = category_count
        self.query_len = query_len
        self.mask_zero = mask_zero
        self.bidirection = bidirection
        self.shared = shared

    def get_model(self):
        """
        创建双塔CNN模型
        :return:
            model:
        """
        input_category = Input(shape=(self.category_count,))
        input_query1 = Input(shape=(self.query_len,))
        input_query2 = Input(shape=(self.query_len,))

        # Layer1: 特征抽取层
        if self.shared:
            # 调用了1次Model，是双塔共享模式
            model = self.__build_feature

            model_1 = model(input_query1)
            model_2 = model(input_query2)
            merge_layers = Concatenate()([model_1.output, model_2.output, input_category])

        else:
            # 调用了2次Model,是双塔非共享模型
            input1 = self.__build_feature(input_query1)
            input2 = self.__build_feature(input_query2)
            merge_layers = Concatenate()([input1.output, input2.output, input_category])

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

    def __build_feature(self, query):
        """
        建立从Embedding到RNN的特征抽取模型部分

        :param
            query: Input of query
        :return:
            model:
        """
        emb = Embedding(input_dim=len(self.word2idx),
                        output_dim=len(self.emb_matrix[0]),
                        weights=[self.emb_matrix], trainable=False, mask_zero=True,
                        input_length=self.query_len)

        model = Sequential()
        model.add(emb)
        for i in range(len(self.hidden_units)):
            if self.bidirection:
                model.add(Bidirectional(LSTM(units=self.hidden_units[i], return_sequences=True)))
            else:
                model.add(LSTM(units=self.hidden_units[i], return_sequences=True))
        return model
