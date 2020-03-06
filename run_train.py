import os

os.environ.setdefault("TF_KERAS", "1")  # 配置bert4keras的keras为tf.keras

from data_process import category_OneHotEncoder
from data_process.dnn_DataLoader import LoadData
from data_process.bert_DataLoader import BertDataGenerator
from data_process.siamesebert_DataLoader import SiameseDataGenerator
from model import SiameseCnnModel, SiameseRnnModel, SiameseBertModel, BertModel
from utils import logger_init, Evaluator, cal_acc

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
import pandas as pd
import argparse

from bert4keras.backend import set_gelu

# 初始化logging
logger = logger_init()

MODEL_CLASS = {"siamese_CNN": SiameseCnnModel,
               "siamese_RNN": SiameseRnnModel,
               "siamese_bert": SiameseBertModel,
               "albert": BertModel}


def train(args):
    if "bert" in args.model_type:
        set_gelu("tanh")  # 切换gelu版本

        # Step1: Load Data
        data_generator = None
        if "siamese" in args.model_type:
            data_generator = BertDataGenerator
        elif "albert" in args.model_type:
            data_generator = SiameseDataGenerator

        train_ds = data_generator(data_path=args.train_data_path, batch_size=args.batch_size,
                                  dict_path=args.bert_dict_path, maxlen=args.query_len)
        dev_ds = data_generator(data_path=args.dev_data_path, batch_size=args.batch_size,
                                maxlen=args.query_len, dict_path=args.bert_dict_path)
        test_ds = data_generator(data_path=args.test_data_path, batch_size=args.batch_size,
                                 maxlen=args.query_len, dict_path=args.bert_dict_path)

        # Step2: Load Model
        model = None
        if "siamese" in args.model_type:
            model = SiameseBertModel(config_path=args.bert_config_path, checkpoint_path=args.bert_checkpoint_path,
                                     dense_units=args.dense_units)
        elif "albert" in args.model_type:
            model = BertModel(config_path=args.bert_config_path, checkpoint_path=args.bert_checkpoint_path)

        model_name = model.__class__.__name__
        model = model.get_model()

        from bert4keras.optimizers import Adam
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(2e-5),  # 用足够小的学习率
            # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
            metrics=['accuracy'],
        )

        evaluator = Evaluator(dev_ds=dev_ds, model_name=model_name, is_bert_model=True)
        logger.info("***** Running training *****")
        logger.info("  Model Class Name = %s", model_name)
        logger.info("  Num Epochs = %d", args.epoch)
        model.fit_generator(train_ds.forfit(),
                            steps_per_epoch=len(train_ds),
                            epochs=args.epoch,
                            callbacks=[evaluator])

        model = load_model('./checkpoints/best_{}.h5'.format(model_name))
        logger.info("***** Test Reslt *****")
        logger.info("  Model = %s", model_name)
        logger.info("  Batch Size = %d", args.batch_size)
        logger.info("  Final Test Acc:%05f", cal_acc(data=test_ds, model=model, is_bert_model=True))

    elif "NN" in args.model_type:
        # Step 1 : Loda Data
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

        dev_ds = loader.dataset(encoder=category_encoder, data_df=dev_data)
        dev_ds = dev_ds.batch(batch_size=args.batch_size)
        test_ds = loader.dataset(encoder=category_encoder, data_df=test_data)
        test_ds = test_ds.batch(batch_size=args.batch_size)

        # Step2: Load Model
        model = None
        if "siamese_CNN" in args.model_type:
            model = SiameseCnnModel(emb_matrix=emd_matrix, word2idx=word2idx, filters_nums=args.filters_nums,
                                    kernel_sizes=args.kernel_sizes, dense_units=args.dense_units,
                                    label_count=args.label_count, category_count=category_count,
                                    query_len=args.query_len, shared=args.feature_shared,
                                    add_feature=args.add_features)
        elif "siamese_RNN" in args.model_type:
            model = SiameseRnnModel(emb_matrix=emd_matrix, word2idx=word2idx, hidden_units=args.hidden_units,
                                    dense_units=args.dense_units, label_count=args.label_count,
                                    category_count=category_count, query_len=args.query_len,
                                    mask_zero=args.mask_zero, bidirection=args.bi_direction,
                                    shared=args.feature_shared, add_feature=args.add_features)
        model_name = model.__class__.__name__
        model = model.get_model()

        logger.info("***** Running training *****")
        logger.info("  Model Class Name = %s", model_name)
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.epoch)

        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["acc"])
        early_stopping = EarlyStopping(monitor="val_acc", patience=3, mode="max")
        evaluator = Evaluator(dev_ds=dev_ds, model_name=model_name, is_bert_model=False, dev_label=dev_data['label'])

        # Step3: Train Model
        history = model.fit(train_ds, callbacks=[early_stopping, evaluator], epochs=args.epoch,
                            steps_per_epoch=len(train_data) // args.batch_size,
                            validation_data=dev_ds, validation_steps=len(dev_data) // args.batch_size)

        # Step4 : Save model and trainLogs
        logger.info("***** Training Logs *****")

        for epoch in history.epoch:
            logger.info("Epoch %d", epoch)
            logger.info("train_loss:%f train_acc:%f val_loss:%f val_acc:%f",
                        history.history.get("loss")[epoch], history.history.get("acc")[epoch],
                        history.history.get("val_loss")[epoch], history.history.get("val_acc")[epoch])
        #
        # time_stamp = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
        # path = './checkpoints/{}_{}.h5'.format(model_name, time_stamp)
        # model.save(path)

        model = load_model('./checkpoints/best_{}.h5'.format(model_name))
        y_pred = model.predict(test_ds)
        y_true = test_data["label"].values.reshape((-1, 1))

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        logger.info("***** Pramaters *****")
        logger.info("  ModelName = %s", args.model_type)
        logger.info("  Add Features = %s", args.add_features)
        logger.info("  Embedding dims = %d", len(emd_matrix[0]))
        logger.info("  BatchSize = %d", args.batch_size)

        if "CNN" in args.model_type:
            logger.info("  kernel_sizes = %s", args.kernel_sizes)
            logger.info("  filters_nums = %s", args.filters_nums)
        elif "RNN" in args.model_type:
            logger.info("  hidden_units = %s", args.hidden_units)
            logger.info("  bi_direction = %s", args.bi_direction)

        logger.info("  dense_units = %s", args.dense_units)
        logger.info("  feature_shared = %s", args.feature_shared)
        logger.info("***** Testing Results *****")
        logger.info("  Acc = %f", acc)
        logger.info("  Precision = %f", precision)
        logger.info("  Recall = %f", recall)
        logger.info("  F1-score = %f", f1)


def main():
    parser = argparse.ArgumentParser()
    # Choose Model & Input
    parser.add_argument("--model_type", type=str, default="siamese_CNN",
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASS.keys()))
    parser.add_argument("--feature_shared", type=str, default="True",
                        help="whether share the feature-struct in simeseNet")
    parser.add_argument("--query_len", type=int, default=40,
                        help="how long of each query in origin data")

    # About w2v_Path
    parser.add_argument("--w2v_path", type=str, default="./w2v/w2v_word_300.pkl",
                        help="path of w2v")
    # About data_path
    parser.add_argument("--train_data_path", type=str, default="./jupyter/shuffle-data/train_data.csv",
                        help="")
    parser.add_argument("--dev_data_path", type=str, default="./jupyter/shuffle-data/dev_data.csv",
                        help="")
    parser.add_argument("--test_data_path", type=str, default="./jupyter/shuffle-data/test_data.csv",
                        help="")
    # Dense Layer
    parser.add_argument("--dense_units", type=str, default="256,64,16",
                        help="units in each dense layer")
    parser.add_argument("--label_count", type=int, default=2,
                        help="how many label to predict")
    # About Train
    parser.add_argument("--batch_size", type=int, default=128,
                        help="how many samples in each batch")
    parser.add_argument("--epoch", type=int, default=30,
                        help="")
    # NN-features
    parser.add_argument("--add_features", type=str, default='True',
                        help="whether to add features for NN")
    # CNN
    parser.add_argument("--kernel_sizes", type=str, default='3,4,5',
                        help="filter sizes to use for convolution")
    parser.add_argument("--filters_nums", type=str, default="32,64,128",
                        help="filter nums for convolution")
    # RNN
    parser.add_argument("--hidden_units", type=str, default='64,64,64',
                        help="how many units in each step for RNN")
    parser.add_argument("--mask_zero", type=str, default='True',
                        help="whether to mask padding in Embedding")
    parser.add_argument("--bi_direction", type=str, default='True',
                        help="whether to build bi-direction features")
    # Bert
    parser.add_argument("--bert_dict_path", type=str,
                        default="./bert_pretrained/albert_tiny_zh_google/vocab.txt",
                        help="")
    parser.add_argument("--bert_config_path", type=str,
                        default="./bert_pretrained/albert_tiny_zh_google/albert_config_tiny_g.json",
                        help="")
    parser.add_argument("--bert_checkpoint_path", type=str,
                        default="./bert_pretrained/albert_tiny_zh_google/albert_model.ckpt",
                        help="")
    args = parser.parse_args()

    # CNN
    args.kernel_sizes = [int(size) for size in str(args.kernel_sizes).split(',')]
    args.filters_nums = [int(num) for num in str(args.filters_nums).split(',')]
    # RNN
    args.hidden_units = [int(num) for num in str(args.hidden_units).split(',')]
    # Dense
    args.add_features = True if args.add_features == "True" else False
    args.dense_units = [int(unit) for unit in str(args.dense_units).split(',')]

    args.feature_shared = True if args.feature_shared == "True" else False
    args.mask_zero = True if args.mask_zero == "True" else False
    args.bi_direction = True if args.bi_direction == "True" else False

    train(args)


if __name__ == "__main__":
    main()
