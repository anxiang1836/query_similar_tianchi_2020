#  新冠疫情相似句对判定大赛-TianChi

比赛链接为：https://tianchi.aliyun.com/competition/entrance/231776/introduction

## 比赛介绍

本次比赛达摩院联合医疗服务机构妙健康发布疫情相似句对判定任务。比赛整理近万条真实语境下疫情相关的肺炎、支原体肺炎、支气管炎、上呼吸道感染、肺结核、哮喘、胸膜炎、肺气肿、感冒、咳血等患者提问句对，要求选手通过自然语言处理技术识别相似的患者问题。

## 项目内容

作为自己在相似度匹配任务上的入门。从基本的双塔模型到bert的fintune，分步实现文本匹配的各种深度模型。

现已实现的模型：

- SiameseCNN
- SiameseRNN
- keras4bert

待实现的模型：

- K-nrm

待实现功能：

- Siamese中多样化考虑提取特征后的处理

- bert后，Dense前再接入其他特征

## 项目结构

```bash
.
├── jupyter       # 数据准备的notebook
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
├── model         # 训练模型
│   ├── __init__.py
│   ├── 00-TFIDF_LR.ipynb
│   ├── basic_model.py # siamese模型的父类
│   ├── siamese_cnn.py
│   ├── siamese_rnn.py
│   └── bert.py 
├── utils           # 工具类
│   ├── __init__.py
│   └── logConfig.py ## Logging配置
├── w2v             # w2v训练notebook      
│   ├── train_w2v.ipynb
│   ├── w2v_char_300.pkl
│   └── w2v_word_300.pkl
├── logs            # 用于存储训练过程的Log
│   └── README.md
├── checkpoints     # 用于存储训练的模型.h5
│   └── README.md
├── bert-pretrained # 用于存储预训练的bert模型
│   └── README.md
├── tcdata
│   └── test.csv
├── requirements.txt
├── Dockerfile 
├── run_predict.py
├── run_train.py
├── run.sh             # 用于生成predict.csv的脚本
├── run_siameseCNN.sh  # CNN模型训练脚本
├── run_siameseRNN.sh  # RNN模型训练脚本
└── run_albert.sh      # albert模型训练脚本
```

