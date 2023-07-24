## This is the source code of IJCNN 2023 paper TieFake: Title-Text Similarity and Emotion-Aware Fake News Detection (TieFake).
# Start

To run the code in this repo, you need to have `Python>=3.9.6`, `PyTorch>=1.9.0`
Other dependencies can be installed using the following commands:

pip install -r requirements.txt
download datasets
clean datasets and save them into folder Data,such as:
--Data
  --politifact_images
    --xx.jpg
    ......
  --gossipcop_images
    --xx.jpg
    ......
  --politifact_train.tsv
  --politifact_test.tsv
  --gossipcop_train.tsv
  --gossipcop_test.tsv

run bert_training.py to train bert in our datasets
run resnest101_training.py to train resnest_101 in our datasets
run main.py to train fusion_model

# Datasets

Complete dataset cannot be distributed because of Twitter privacy policies and news publisher copy rights. The dataset includes fake&real from dataset [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet),including [Politifact](https://www.politifact.com/) and [Gossipcop](https://www.gossipcop.com/).

After we clean the datasets, the statistics of the dataset is shown below:

| News Articles  | #Fake News| #True News | #Total News |
| Politifact     |  161      |   205      |   366       |
| Gossipcop      |  4927     |   11693    |   21620     |

If you use the code in your project, please cite the following paper:
IJCNN'23 ([PDF](https://arxiv.org/pdf/2304.09421.pdf))
```bibtex
@inproceedings{guo2023TieFake,
  title={TieFake: Title-Text Similarity and Emotion-Aware Fake News Detection},
  author={Guo, Quanjiang and Kang, Zhao and Tian, Ling and Chen, Zhouguo},
  booktitle={Proceedings of the IEEE International Joint Conference on Neural Networks 2023},
  year={2023}
}
```
Please email to guochance1999@163.com for other inquiries.
