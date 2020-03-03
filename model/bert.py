from bert4keras.bert import build_bert_model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Model


class BertModel:
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        初始化预训练的模型参数
        :param config_path    :
        :param checkpoint_path:
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def get_model(self):
        # 加载预训练模型
        bert = build_bert_model(
            config_path=self.config_path, checkpoint_path=self.checkpoint_path,
            with_pool=True, return_keras_model=False, model="albert")

        output = Dropout(rate=0.1)(bert.model.output)
        output = Dense(units=2,
                       activation='softmax',
                       kernel_initializer=bert.initializer)(output)

        model = Model(bert.model.input, output)
        model.summary()
        return model
