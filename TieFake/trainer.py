import json
import time
import copy
from collections import deque

from tqdm import tqdm
import torch
from statistics import mean
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score,precision_score,recall_score
from sklearn.utils.multiclass import type_of_target
from numpyencoder import NumpyEncoder



class ModelTrainer:

    def __init__(self, data_types: list, datasets: dict, dataloaders: dict, model: torch.nn.Module):
        assert isinstance(datasets, dict)
        for datatype in data_types:
            assert datatype in datasets and datatype in dataloaders, "Missing dataset or dataloader"
        self._l_datatypes = data_types
        self._datasets = datasets
        self._dataloaders = dataloaders
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Current device: {self.device}')

    def save_model(self, path):
        print(f"saving model to {path}")
        torch.save(self.model.state_dict(), path)

    def train_model(self, criterion, optimizer, scheduler, num_epochs=2, report_len=500):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        dataset_sizes = {x: len(self._datasets[x]) for x in self._l_datatypes}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in self._l_datatypes:
                print(f'{phase} phase')
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval() 
                running_loss = 0.0
                running_corrects = 0

                loss_q = deque(maxlen=report_len)
                acc_q = deque(maxlen=report_len)
                counter = 0
                for inputs in tqdm(self._dataloaders[phase]):
                    counter += 1
                    print(inputs)
                    inputs = {x: inputs[x].to(self.device) for x in inputs}
                    labels = inputs['label']
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        t_pred = outputs > 0.5
                        acc = (t_pred.squeeze() == labels).float().sum() / len(labels)
                        acc_q.append(acc.item())
                        loss = criterion(outputs, labels.unsqueeze(-1).float())
                        loss_q.append(loss.item())
                        if counter % report_len == 0:
                            print(f"Iter {counter}, loss: {mean(loss_q)}, accuracy:{mean(acc_q)}")
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs['image'].size(0)
                    running_corrects += torch.sum(t_pred.squeeze() == labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        self.model.load_state_dict(best_model_wts)

    def generate_eval_report(self, json_path):

        assert 'test' in self._l_datatypes, 'test set not available!'
        self.model.eval()
        l_pred, l_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self._dataloaders['test']):
                batch = {x: batch[x].to(self.device) for x in batch}
                bat_labels = batch['label']
                bat_out = self.model(batch)
                t_pred = bat_out > 0.5
                l_pred.append(t_pred.squeeze().cpu().numpy())
                l_labels.append(bat_labels.squeeze().cpu().numpy())
        metrics = {
            'cmat': confusion_matrix(l_labels, l_pred),
            'f1': f1_score(l_labels, l_pred),
            'mcc': matthews_corrcoef(l_labels, l_pred),
            'accuracy': accuracy_score(l_labels, l_pred)
        }
        with open(json_path, 'w') as j:
            json.dump(metrics, j, cls=NumpyEncoder)