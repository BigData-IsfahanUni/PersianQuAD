<span align="center">
    <a href="https://www.kaggle.com/jamshidjdmy/persianquad"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=PersianQuAD&logo=Kaggle&color=20BEFF"/></a>
    <a href="https://colab.research.google.com/github/jamshidjdmy/PersianQuAD/blob/main/Main.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Model&logo=Google%20Colab&color=f9ab00"></a>
</span>

# PersianQuAD: The Native Question Answering Dataset for the Persian Language
In order to address the need for a high-quality QA dataset for Persian language, we propose a model for creating dataset for deep-learning-based QA systems. We deploy the proposed model to create PersianQuAD, the first native question answering dataset for the Persian language. PersianQuAD contains approximately 20,000 "question, paragraph, answer" triplets on [Persian Wikipedia](https://fa.wikipedia.org/) articles and is the first large-scale native QA dataset for the Persian language which is created by native annotators.

The proposed model consists of four steps: 1) Wikipedia article selection, 2) question-answer collection, 3) three-candidates test set preparation, and 4) Data Quality Monitoring.  We analysed PersianQuAD and showed that it contains questions of varying types and difficulties and hence, it is a good presenter of real-world questions in the Persian language. We built three QA systems using MBERT, ALBERT-FA and ParsBERT. The best system uses MBERT and achieves a F1 score of 82.97% and an Exact Match of 78.8%. The results show that the resulted dataset performs well for training deep-learning-based QA systems. We have made our dataset and QA models freely available and hope that it encourages the development of new QA datasets and systems for different languages, and leads to further advances in machine comprehension.
# Dataset
### Download
The dataset is available for download from the [`Dataset`](https://github.com/JamshidJDMY/PersianQuAD/tree/main/Dataset) directory. The statistics of the PersianQuAD is shown below:
| Split | No. of questions | No. of Candidate Answers | Avg. of question length |  avg. answer length   |
| :---: |  :------------:  |    :----------------:    |  :------------------:   | :-------------------: |
| Train |      18567       |            1             |          10.7           |          2.6          |
| Test  |       1000       |            3             |          10.5           |          2.3          |

In the following, question type distribution over PersianQuAD dataset is illustrated:
| Question Word | Distribution |
| :-----------: | :----------: |
|      What     |    28.14%    |
|      How      |    15.24%    |
|      When     |    10.70%    |
|     Where     |    13.60%    |
|      Who      |    16.50%    |
|     Which     |    15.26%    |
|      Why      |    00.92%    |
# Model
You can train and test the proposed model by running `Main.ipynb` in the Google Colab enviroment. You must download the [`repository`](https://github.com/JamshidJDMY/PersianQuAD/archive/refs/heads/main.zip) and extract it to your Google Drive. Then, run `Main.ipynb` by Google Colab and train your models.
# Evalution
We build three QA systems according to the pre-trained language models examined (MBERT, ALBERT-FA, ParsBERT). We trained each of the QA systems using the training part of PersianQuAD and evaluate them using the test part. We evaluate each of the QA systems according to two widely used automatic evaluation metrics *Exact Match* and *F1*.
|   Dataset   |   Model   | Exact Match | F1 measure |
| :---------: | :-------: | :---------: | :--------: |
| PersianQuAD |   Human   |    95.00%   |   96.49%   |
| PersianQuAD | Albert-FA |    74.90%   |   79.25%   |
| PersianQuAD |  ParsBERT |    73.80%   |   79.08%   |
| **PersianQuAD** |   **MBERT**   | **78.80%**  | **82.97%** |

# Citation
The paper will be published a few days later...
