#  新冠疫情相似句对判定大赛-TianChi

比赛链接为：https://tianchi.aliyun.com/competition/entrance/231776/introduction

## 比赛介绍

本次比赛达摩院联合医疗服务机构妙健康发布疫情相似句对判定任务。比赛整理近万条真实语境下疫情相关的肺炎、支原体肺炎、支气管炎、上呼吸道感染、肺结核、哮喘、胸膜炎、肺气肿、感冒、咳血等患者提问句对，要求选手通过自然语言处理技术识别相似的患者问题。

## 项目内容

作为自己在相似度匹配任务上的入门。从基本的双塔模型到bert的fintune，分步实现文本匹配的各种深度模型。

### 实现模型

- SiameseCNN
- SiameseRNN
- albert(bert4keras)[sentence pairs]
- SiameseBert

### 数据增强

根据`IF A=B and A=C THEN B=C`的规则，对正样本做了扩充增强。

### 特征工程

在SiameseNN中，构建了5种特征（即考虑多种距离测量的方式）：

- query1，query2
- |query1 - query2|
- query1 \* query2
- max(query1,query2)^2
- category（OneHot表示）

### 模型结果（带补充）

## 项目结构

```bash
.
├── jupyter       # 数据准备的notebook
│   ├── EDA.ipynb
│   ├── Augmentation.ipynb
│   ├── origin-data   # 原始数据
│   ├── shuffle-data  # 预处理并切分后数据
│   └── augment-data  # 数据增强后数据
├── data_process  # 数据加载模块
│   ├── __init__.py
│   ├── dnn_DataLoader.py
│   └── bert_DataLoader.py
├── model         # 训练模型
│   ├── __init__.py
│   ├── 00-TFIDF_LR.ipynb
│   ├── basic_model.py # siamese模型的父类
│   ├── siamese_cnn.py
│   ├── siamese_rnn.py
│   └── bert.py 
├── utils           # 工具类
│   ├── __init__.py
│   ├── evaluate.py
│   └── logConfig.py ## Logging配置
├── w2v             # w2v训练notebook      
│   ├── train_w2v.ipynb
│   └── ...
├── logs            # 用于存储训练过程的Log
│   └── ...
├── checkpoints     # 用于存储训练的模型.h5
│   └── ...
├── bert-pretrained # 用于存储预训练的bert模型
│   └── ...
├── requirements.txt
├── Dockerfile 
├── run_predict.py
├── run_train.py
├── run.sh
├── run_albert.sh
├── run_siameseCNN.sh
└── run_siameseRNN.sh
```

