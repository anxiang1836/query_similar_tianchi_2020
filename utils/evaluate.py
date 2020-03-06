from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score
import datetime


class Evaluator(Callback):
    """
    用于判定并保存val-acc最好的一组参数，
    """

    def __init__(self, dev_ds, model_name, is_bert_model, dev_label=None):
        super().__init__()
        self.best_val_acc = 0.
        self._dev_ds = dev_ds
        self._model_name = model_name
        self.is_bert_model = is_bert_model
        self._dev_label = dev_label

    def on_epoch_end(self, epoch, logs=None):
        val_acc = cal_acc(data=self._dev_ds, model=self.model, is_bert_model=self.is_bert_model,
                          label=self._dev_label)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # time_stamp = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            self.model.save('./checkpoints/best_{}.h5'.format(self._model_name))
        # test_acc = cal_acc(self._test_ds, self.model)
        # print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
        #       % (val_acc, self.best_val_acc, test_acc))


def cal_acc(data, model, is_bert_model=True, label=None):
    if is_bert_model:
        # bert的数据是list的形式装载的
        total = 0.
        right = 0.
        for x, y_true in data:
            y_pred = model.predict(x).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total

    else:
        # NN的数据是DataSet的形式装载的
        y_true = label.values.reshape((-1, 1))

        y_pred = model.predict(data)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        return accuracy_score(y_true, y_pred)
