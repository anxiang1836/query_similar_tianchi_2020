from data_process import category_OneHotEncoder
from data_process.dnn_DataLoader import LoadData
import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.models import load_model


def predict(args):
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv("./tcdata/test.csv")
    category_encoder = category_OneHotEncoder(data_df=train_data)

    loader = LoadData(w2v_path=args.w2v_path, query_len=args.query_len)
    test_ds = loader.dataset(encoder=category_encoder, data_df=test_data).batch(batch_size=args.batch_size)

    model = load_model(args.saved_model_path)

    y_pred = model.predict(test_ds)  # type:np.ndarray
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    result_df = pd.DataFrame({"id": test_data["id"], "label": y_pred.flatten()})
    result_df["label"] = result_df["label"].map(lambda x: int(x))
    result_df.to_csv(args.result_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_path", type=str, default="./w2v/w2v_word_300.pkl",
                        required=True, help="path of w2v")
    parser.add_argument("--train_data_path", type=str, default="./jupyter/shuffle-data/train_data.csv",
                        required=True, help="")
    parser.add_argument("--saved_model_path", type=str, default="./checkpoints/SimeseCnnModel_03-01_15-52-57.h5",
                        required=True, help="path of trained_model")
    parser.add_argument("--batch_size", type=int, default=32,
                        required=True, help="how many samples in each batch, Same with train")
    parser.add_argument("--query_len", type=int, default=40,
                        required=True, help="how long of each query in origin data, Same with train")

    parser.add_argument("--result_path", type=str, default="./result.csv",
                        help="path of w2v")

    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()
