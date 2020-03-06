from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.tokenizer import Tokenizer
import numpy as np


class SiameseDataGenerator(DataGenerator):
    """
    SiameseBert的数据生成器，生成的数据组成为：
    """

    def __init__(self, data_path: str, batch_size: int, maxlen: int, dict_path: str):
        super().__init__(data=self.__load_data(data_path), batch_size=batch_size)
        self._tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self._maxlen = maxlen

    @staticmethod
    def __load_data(filename: str):
        D = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                category, text1, text2, label = line.strip().split(',')
                if category != 'category':
                    # 过滤掉columns数据行
                    D.append((text1, text2, int(label)))
        return D

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        q1_batch_token_ids, q1_batch_segment_ids, q2_batch_token_ids, q2_batch_segment_ids, \
        batch_labels = [], [], [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            q1_token_ids, q1_segment_ids = self._tokenizer.encode(text1, max_length=self._maxlen)
            q2_token_ids, q2_segment_ids = self._tokenizer.encode(text2, max_length=self._maxlen)

            q1_batch_token_ids.append(q1_token_ids)
            q2_batch_token_ids.append(q2_token_ids)
            q1_batch_segment_ids.append(q1_segment_ids)
            q2_batch_segment_ids.append(q2_segment_ids)
            batch_labels.append([label])

            if len(batch_labels) == self.batch_size or i == idxs[-1]:
                q1_batch_token_ids = sequence_padding(q1_batch_token_ids)
                q2_batch_token_ids = sequence_padding(q2_batch_token_ids)

                q1_batch_segment_ids = sequence_padding(q1_batch_segment_ids)
                q2_batch_segment_ids = sequence_padding(q2_batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [q1_batch_token_ids, q1_batch_segment_ids, q2_batch_token_ids, q2_batch_segment_ids], batch_labels

                q1_batch_token_ids, q1_batch_segment_ids, q2_batch_token_ids, q2_batch_segment_ids, \
                batch_labels = [], [], [], [], []
