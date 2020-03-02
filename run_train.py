from data_process import category_OneHotEncoder, LoadData
from model import SimeseCnnModel
from utils import logger_init

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import datetime
import argparse

# 初始化logging
logger = logger_init()
MODEL_TYPE = {"simese_CNN_shared": SimeseCnnModel}


def train(args):
    # Step1: Load Data
    train_data = pd.read_csv(args.train_data_path)
    dev_data = pd.read_csv(args.dev_data_path)
    test_data = pd.read_csv(args.test_data_path)

    category_count = len(train_data["category"].value_counts())
    category_encoder = category_OneHotEncoder(data_df=train_data)

    loader = LoadData(w2v_path=args.w2v_path, query_len=args.query_len)
    word2idx = loader.word2idx
    emd_matrix = loader.emb_matrix

    """
    注意：
    shuffle的顺序很重要:一般建议是先执行shuffle方法，接着采用batch方法。
    这样是为了保证在整体数据打乱之后再取出batch_size大小的数据。
    如果先采取batch方法再采用shuffle方法，那么此时就只是对batch进行shuffle，
    而batch里面的数据顺序依旧是有序的，那么随机程度会减弱。
    """
    train_ds = loader.dataset(encoder=category_encoder, data_df=train_data)
    train_ds = train_ds.shuffle(buffer_size=len(train_data)).batch(batch_size=args.batch_size).repeat()

    dev_ds = loader.dataset(encoder=category_encoder, data_df=dev_data).batch(batch_size=args.batch_size)
    test_ds = loader.dataset(encoder=category_encoder, data_df=test_data).batch(batch_size=args.batch_size)

    # Step2: Load Model
    model = MODEL_TYPE[args.model_type]
    model = model(emb_matrix=emd_matrix, word2idx=word2idx, filters_nums=args.filters_nums,
                  kernel_sizes=args.kernel_sizes, dense_units=args.dense_units, label_count=args.label_count,
                  category_count=category_count, query_len=args.query_len, shared=args.feature_shared)

    model_name = model.__class__.__name__
    model = model.get_model()

    logger.info("***** Running training *****")
    logger.info("  Model Class Name = %s", model_name)
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Num BatchSize = %d", args.batch_size)

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["acc"])
    early_stopping = EarlyStopping(monitor="val_acc", patience=3, mode="max")

    # Step3: Train Model
    history = model.fit(train_ds, callbacks=[early_stopping], epochs=args.epoch,
                        steps_per_epoch=len(train_data) // args.batch_size,
                        validation_data=dev_ds, validation_steps=len(dev_data) // args.batch_size)

    # Step4 : Save model and trainLogs
    logger.info("***** Training Logs *****")

    for epoch in history.epoch:
        logger.info("Epoch %d", epoch)
        logger.info("train_loss:%f train_acc:%f val_loss:%f val_acc:%f",
                    history.history.get("loss")[epoch], history.history.get("acc")[epoch],
                    history.history.get("val_loss")[epoch], history.history.get("val_acc")[epoch])

    time_stamp = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    path = './checkpoints/{}_{}.h5'.format(model_name, time_stamp)
    model.save(path)

    y_pred = model.predict(test_ds)
    y_true = test_data["label"].values.reshape((-1, 1))

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logger.info("***** Pramaters *****")
    logger.info("  SavedModelPath = %s", path)
    logger.info("  BatchSize = %d", args.batch_size)
    logger.info("  kernel_sizes = %s", args.kernel_sizes)
    logger.info("  filters_nums = %s", args.filters_nums)
    logger.info("  dense_units = %s", args.dense_units)
    logger.info("  feature_shared = %s", args.feature_shared)
    logger.info("***** Testing Results *****")
    logger.info("  Acc = %f", acc)
    logger.info("  Precision = %f", precision)
    logger.info("  Recall = %f", recall)
    logger.info("  F1-score = %f", f1)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="simese_CNN_shared",
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPE.keys()))
    parser.add_argument("--w2v_path", type=str, default="./w2v/w2v_word_300.pkl",
                        help="path of w2v")
    parser.add_argument("--train_data_path", type=str, default="./jupyter/shuffle-data/train_data.csv",
                        help="")
    parser.add_argument("--dev_data_path", type=str, default="./jupyter/shuffle-data/dev_data.csv",
                        help="")
    parser.add_argument("--test_data_path", type=str, default="./jupyter/shuffle-data/test_data.csv",
                        help="")
    parser.add_argument("--kernel_sizes", type=str, default='3,4,5',
                        help="filter sizes to use for convolution")
    parser.add_argument("--filters_nums", type=str, default="32,64,128",
                        help="filter nums for convolution")
    parser.add_argument("--dense_units", type=str, default="256,64,16",
                        help="units in each dense layer")
    parser.add_argument("--label_count", type=int, default=2,
                        help="how many label to predict")
    parser.add_argument("--query_len", type=int, default=40,
                        help="how long of each query in origin data")
    parser.add_argument("--feature_shared", type=str, default="True",
                        help="whether share the feature-struct in simeseNet")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="how many samples in each batch")
    parser.add_argument("--epoch", type=int, default=30,
                        help="")

    args = parser.parse_args()

    args.kernel_sizes = [int(size) for size in str(args.kernel_sizes).split(',')]
    args.filters_nums = [int(num) for num in str(args.filters_nums).split(',')]
    args.dense_units = [int(unit) for unit in str(args.dense_units).split(',')]
    if args.feature_shared == "True":
        args.feature_shared = True
    else:
        args.feature_shared = False

    train(args)


if __name__ == "__main__":
    main()
