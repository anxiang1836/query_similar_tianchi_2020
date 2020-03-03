from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.tokenizer import Tokenizer
import numpy as np


class Data_generator(DataGenerator):
    """数据生成器
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
                id, category, text1, text2, label = line.strip().split(',')
                if category != 'category':
                    D.append((text1, text2, int(label)))
        return D

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            token_ids, segment_ids = self._tokenizer.encode(text1, text2, max_length=self._maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
