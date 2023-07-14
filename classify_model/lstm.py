import os
import torch, pickle
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig

from classify_model.dataset import TraditionalDataset
from classify_model.score import get_MCM_score
from classify_model.ce import CE_CodeBert, load_data
from classify_model.traditional import LSTM, GRU, CNN, LSTM_Atten
from prettytable import PrettyTable

def sava_data(filename, data):
    print("开始保存数据至于：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

class CE_LSTM(CE_CodeBert):
    def __init__(self, max_len=256, n_classes=2, epochs=15,model_path = 'microsoft/codebert-base',
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = LSTM(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        

    def preparation(self, X_train,  y_train, X_valid, y_valid, train_df):
        # create datasets
        self.train_df = train_df
        self.embeddings = load_data("/mnt/qm_data/small-ase2022/data/mvd_allkfold_40_word2vec_embedding.pkl")
        # self.embeddings = load_data("/mnt/qm_data/small-ase2022/data/mvd_allkfold_40_word2vec_nosub_embedding.pkl")
        self.train_set = TraditionalDataset(X_train, y_train, self.max_len, self.embeddings)
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.embeddings)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs,_  = self.model( vectors )
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))   # 获取预测
            labels += list(np.array(targets.cpu()))      # 获取标签

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0

        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, _ = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict


##########################################################################################################

class CE_LSTM_Atten(CE_LSTM):
    def __init__(self, max_len=256, n_classes=2, epochs=15,model_path = 'microsoft/codebert-base',
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = LSTM_Atten(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)

##########################################################################################################

class CE_GRU(CE_LSTM):
    def __init__(self, max_len=256, n_classes=2, epochs=15,model_path = 'microsoft/codebert-base',
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = GRU(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)


##########################################################################################################

class CE_CNN(CE_LSTM):
    def __init__(self, max_len=256, n_classes=2, epochs=15,model_path = 'microsoft/codebert-base',
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = CNN(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)