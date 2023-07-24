import sys
import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms
from transformers import BertForSequenceClassification
from resnest101 import resnest101_2way
from dataloader import Hybrid_Dataset, my_collate
from models import FusionModel
from trainer import ModelTrainer

bert_classifier = BertForSequenceClassification.from_pretrained('./bert_save_dir')
resnest_model = resnest101_2way(pretrained=False)
resnest_dict = torch.load('./resnest101_epochs10_full_train.pt')
resnest_model.load_state_dict(resnest_dict)

hybrid_model = FusionModel(resnest_model, bert_classifier)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hybrid_model = hybrid_model.to(device)

csv_dir = "./Data/"
img_dir = "./Data/gossipcop_images/"
l_datatypes = ['train', 'val', 'test']
csv_fnames = {
    'train': 'gossipcop_train.tsv',
    'val': 'gossipcop_test.tsv',
    'test': 'gossipcop_test.tsv'
}
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
if __name__ == '__main__':
    hybrid_datasets = {x: Hybrid_Dataset(os.path.join(csv_dir, csv_fnames[x]), img_dir, transform=data_transforms)
                    for x in l_datatypes}
    dataset_sizes = {x: len(hybrid_datasets[x]) for x in l_datatypes}

    dataloaders = {x: torch.utils.data.DataLoader(hybrid_datasets[x], batch_size=16, shuffle=True, num_workers=2,
                                                collate_fn=my_collate) for x in l_datatypes}
    criterion = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.Adam(hybrid_model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    trainer = ModelTrainer(l_datatypes, hybrid_datasets, dataloaders, hybrid_model)

    trainer.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10, report_len=1000)
    trainer.save_model('./result/hybrid_model.pt')
    trainer.generate_eval_report('./result/hybrid_report.json')