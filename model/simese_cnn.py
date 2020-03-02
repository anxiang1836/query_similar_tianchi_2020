from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, Concatenate, \
    BatchNormalization, Activation, Dense
import numpy as np
from typing import List


class SimeseCnnModel(object):
    def __init__(self, emb_matrix: np.ndarray, word2idx: np.ndarray,
                 filters_nums: List[int], kernel_sizes: List[int], dense_units: List[int],
                 label_count: int, category_count: int, query_len: int, shared: bool = True):
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
        self.emb_matrix = emb_matrix
        self.word2idx = word2idx
        if len(filters_nums) == len(kernel_sizes):
            self.filters_nums = filters_nums
            self.kernel_sizes = kernel_sizes
        else:
            raise ValueError("fitlers_nums is not equal to kernel_sizes")

        self.dense_units = dense_units
        self.label_count = label_count
        self.category_count = category_count
        self.query_len = query_len
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
            model = self.__build_feature()

            query_1 = model(input_query1)
            query_2 = model(input_query2)
            merge_layers = Concatenate()([query_1, query_2, input_category])
            # merge_layers = Concatenate()([query_1.output, query_2.output, input_category])

        else:
            # 调用了2次Model,是双塔非共享模型
            query_1 = self.__build_feature()(input_query1)
            query_2 = self.__build_feature()(input_query2)
            merge_layers = Concatenate()([query_1, query_2, input_category])
            # merge_layers = Concatenate()([query_1.output, query_2.output, input_category])

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

    def __build_feature(self) -> Sequential:
        """
        建立从Embedding到CNN的特征抽取模型部分

        :param
            query: Input of query
        :return:
            model:
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
        #     if i == 0:
        #         conv = Conv1D(filters=self.filters_nums[i], kernel_size=self.kernel_sizes[i])(emb)
        #         conv = BatchNormalization()(conv)
        #         conv = Activation(activation="relu")(conv)
        #         conv = MaxPooling1D()(conv)
        #     elif i == len(self.filters_nums) - 1:
        #         conv = Conv1D(filters=self.filters_nums[i], kernel_size=self.kernel_sizes[i])(conv)
        #         conv = BatchNormalization()(conv)
        #         conv = Activation(activation="relu")(conv)
        #         conv = GlobalMaxPool1D()(conv)
        #     else:
        #         conv = Conv1D(filters=self.filters_nums[i], kernel_size=self.kernel_sizes[i])(conv)
        #         conv = BatchNormalization()(conv)
        #         conv = Activation(activation="relu")(conv)
        #         conv = MaxPooling1D()(conv)
        #
        # model = Model(inputs=query, outputs=conv)

        return model
