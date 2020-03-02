from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Flatten
import numpy as np
from typing import List
from model import BasicModel


class SiameseRnnModel(BasicModel):
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
        super().__init__(emb_matrix, word2idx, dense_units, label_count, category_count, query_len, shared)
        self.hidden_units = hidden_units
        self.mask_zero = mask_zero
        self.bidirection = bidirection

    def build_feature(self):
        """
        建立从Embedding到RNN的特征抽取模型部分
        """

        model = Sequential()
        emb = Embedding(input_dim=len(self.word2idx),
                        output_dim=len(self.emb_matrix[0]),
                        weights=[self.emb_matrix], trainable=False, mask_zero=True,
                        input_length=self.query_len)
        model.add(emb)

        for i in range(len(self.hidden_units)):
            if self.bidirection:
                model.add(Bidirectional(LSTM(units=self.hidden_units[i], return_sequences=True)))
            else:
                model.add(LSTM(units=self.hidden_units[i], return_sequences=True))

        model.add(Flatten())
        return model
