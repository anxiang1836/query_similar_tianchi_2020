from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Activation, \
    BatchNormalization
import numpy as np
from typing import List
from model import BasicModel


class SiameseCnnModel(BasicModel):
    def __init__(self, emb_matrix: np.ndarray, word2idx: np.ndarray, filters_nums: List[int], kernel_sizes: List[int],
                 dense_units: List[int], label_count: int, category_count: int, query_len: int, shared: bool = True):
        """
        初始化模型参数

        :param emb_matrix    : embedding矩阵
        :param word2idx      : word转idx的矩阵
        :param filters_nums  : List[int] 从Embedding到Dense的各卷积层的卷积核个数
        :param kernel_sizes  : List[int] 从Embedding到Dense的的各卷积核的大小
        :param dense_units   : List[int] 从Conv层到Output的各Dense层的神经元个数
        :param label_count   : int 待预测的标签数
        :param category_count: int 原始数据中query类型数量
        :param query_len     : int 原始数据中query的最大长度
        :param shared        : 双塔是否共享特征提取的CNN权重
        """
        super().__init__(emb_matrix, word2idx, dense_units, label_count, category_count, query_len, shared)

        if len(filters_nums) == len(kernel_sizes):
            self.filters_nums = filters_nums
            self.kernel_sizes = kernel_sizes
        else:
            raise ValueError("fitlers_nums is not equal to kernel_sizes")

    def build_feature(self) -> Sequential:
        """
        建立从Embedding到CNN的特征抽取模型部分
        """
        emb = Embedding(input_dim=len(self.word2idx),
                        output_dim=len(self.emb_matrix[0]),
                        weights=[self.emb_matrix], trainable=False,
                        input_length=self.query_len)

        model = Sequential()
        model.add(emb)

        for i in range(len(self.filters_nums)):
            model.add(Conv1D(filters=self.filters_nums[i], kernel_size=self.kernel_sizes[i]))
            model.add(BatchNormalization())
            model.add(Activation(activation="relu"))
            if i == len(self.filters_nums) - 1:
                model.add(GlobalMaxPool1D())
            else:
                model.add(MaxPooling1D())

        return model
