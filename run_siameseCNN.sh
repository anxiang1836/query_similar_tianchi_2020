python run_train.py \
  --model_type="siamese_CNN" \
  --add_features='True' \
  --w2v_path='./w2v/w2v_char_100.pkl' \
  --train_data_path='./jupyter/augment-data/train_data.csv' \
  --dev_data_path='./jupyter/augment-data/dev_data.csv' \
  --test_data_path='./jupyter/shuffle-data/test_data.csv' \
  --kernel_sizes='3,4,5' \
  --filters_nums='32,64,128' \
  --dense_units='256,128,32' \
  --label_count=2 \
  --query_len=40 \
  --feature_shared='True' \
  --batch_size=32 \
  --epoch=30