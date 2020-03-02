python run_predict.py \
  --w2v_path='./w2v/w2v_char_300.pkl' \
  --train_data_path='./jupyter/shuffle-data/train_data.csv' \
  --saved_model_path='./checkpoints/SimeseCnnModel_03-01_15-52-57.h5' \
  --batch_size=32 \
  --query_len=40 \
  --result_path='./result.csv'