#  新冠疫情相似句对判定大赛-TianChi

从基本的双塔模型，分步实现文本匹配的各种深度模型。

## 项目结构

```bash
.
├── jupyter   # 数据准备的notebook
│   ├── EDA.ipynb
│   ├── origin-data
│   │   ├── dev.csv
│   │   └── train.csv
│   └── shuffle-data
│       ├── dev_data.csv
│       ├── test_data.csv
│       └── train_data.csv
├── data_process  # 数据预处理
│   ├── __init__.py
│   └── dnn_DataLoader.py
├── w2v    # w2v训练notebook      
│   ├── train_w2v.ipynb
│   ├── w2v_char_300.pkl
│   └── w2v_word_300.pkl
├── model  # 训练模型
│   ├── 00-TFIDF_LR.ipynb
│   ├── __init__.py
│   ├── simese_cnn.py
│   └── simese_rnn.py
├── logs          # 用于存储训练过程的Log
│   ├── 202003021631.log
│   └── README.md
├── checkpoints   # 用于存储训练的模型.h5
│   └── README.md
├── utils         # 工具类
│   ├── __init__.py
│   └── logConfig.py ## Logging配置
├── tcdata
│   └── test.csv
├── requirements.txt
├── Dockerfile
├── run.sh  
├── run_predict.py
├── run_simeseCNN.sh
└── run_train.py



```

