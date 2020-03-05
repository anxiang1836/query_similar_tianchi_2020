from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import jieba
from typing import List
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


class LoadData:
    def __init__(self, w2v_path: str, query_len: int = 40):
        self.w2v_model = Word2Vec.load(w2v_path)  # type:Word2Vec
        self.query_len = query_len  # the max_len of each query
        self._word2idx = {}
        self._emb = None
        self._w2v_type = "char"

        if "char" in w2v_path:
            self._w2v_type = "char"
        elif "word" in w2v_path:
            self._w2v_type = "word"

        self.__init_word_dict()

    def dataset(self, encoder: OneHotEncoder, data_df: pd.DataFrame):
        """
        Create tf.data.DataSet for data_df.
        :param encoder:
            OneHotEncoder for category in data_df
        :param data_df:
            Origin Data which type is pd.DataFrame with columns :["category,query1,query2,label"]

        :return:
            dataset: [[category,query1,query2],label]
        """
        label = data_df["label"].values
        category = encoder.transform(data_df["category"].values.reshape(-1, 1)).toarray()
        query1, query2 = self.__padding(data_df)
        query1 = self.trans2idx(query1)
        query2 = self.trans2idx(query2)

        input_ds = tf.data.Dataset.from_tensor_slices((category, query1, query2))
        output_ds = tf.data.Dataset.from_tensor_slices(label)

        dataset = tf.data.Dataset.zip((input_ds, output_ds))

        return dataset

    def __padding(self, data_df: pd.DataFrame):
        """
        对原始数据进行PADDING处理
        :return:
            query1, query2
        """
        query_len = self.query_len
        w2v_type = self._w2v_type

        query1 = data_df["query1"].values
        query2 = data_df["query2"].values

        def split(query):
            if w2v_type == "char":
                query = [[c for c in line] for line in query]
            elif w2v_type == "word":
                query = [jieba.lcut(line) for line in query]
            return query

        query1 = split(query1)
        query2 = split(query2)

        def insert_pad_token(line: List):
            if len(line) < query_len:
                for i in range(query_len - len(line)):
                    line.append("_PAD")
            return line

        query1 = [insert_pad_token(line) for line in query1]
        query2 = [insert_pad_token(line) for line in query2]
        return query1, query2

    def trans2idx(self, query):
        """
        将原始文本样本，转换为idx序列
        :param
            query: pd.Series 原始dataframe中的query列
        :return:
            idx_matrix: 转换后的matrix
        """
        sample_num = len(query)  # 样本数量
        query_len = self.query_len  # 每条样本的长度
        idx_matrix = np.zeros((sample_num, query_len))

        for row_idx, line in enumerate(query):
            for column_idx, word in enumerate(line):
                if word in self.word2idx.keys():
                    word_idx = self.word2idx.get(word)
                else:
                    word_idx = 1
                idx_matrix[row_idx][column_idx] = word_idx

        return idx_matrix

    def __init_word_dict(self):
        """
        初始化word2idx与emb_matrix
        :return:
        """
        model = self.w2v_model  # type:Word2Vec
        word2idx = {"_PAD": 0, "_UNK": 1}

        embedding_matrix = np.zeros((len(model.wv.vocab) + 2, model.vector_size))

        unk = np.random.random(size=model.vector_size)
        unk = unk - unk.mean()
        embedding_matrix[1] = unk

        for word in model.wv.vocab.keys():
            idx = len(word2idx)
            word2idx[word] = idx
            embedding_matrix[idx] = model.wv[word]

        self._emb = embedding_matrix
        self._word2idx = word2idx

    @property
    def emb_matrix(self):
        return self._emb

    @property
    def word2idx(self):
        return self._word2idx


def category_OneHotEncoder(data_df: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder().fit(data_df["category"].values.reshape(-1, 1))
    return encoder
