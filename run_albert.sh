python run_train.py \
  --model_type="albert" \
  --train_data_path='./jupyter/shuffle-data/train_data.csv' \
  --dev_data_path='./jupyter/shuffle-data/dev_data.csv' \
  --test_data_path='./jupyter/shuffle-data/test_data.csv' \
  --bert_dict_path='./bert_pretrained/albert_tiny_google_zh_489k/vocab.txt'
  --bert_config_path='./bert_pretrained/albert_tiny_google_zh_489k/albert_config.json'
  --bert_checkpoint_path="./bert_pretrained/albert_tiny_google_zh_489k/bert_model.ckpt"
  --batch_size=8 \
  --epoch=20