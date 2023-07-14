
from sklearn import cluster
from classify_model.pairloss import SupConLoss
import torch, pickle
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import torch.nn as nn
from classify_model.dataset import CustomDataset, SCCL_CustomDataset
from classify_model.pairloss import SupConLoss
from classify_model.score import get_MCM_score_with_new_label
from classify_model.ori import Ori_CodeBert
from classify_model.ori_kmeans import Ori_Kmeans
from classify_model.ce import CE_CodeBert
from sklearn.cluster import KMeans

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

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()

class Kmeans(CE_CodeBert):
    def __init__(self, model_path, tokenizer_path, cluster_centers, max_len=256, n_classes=2, epochs=15,
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt',
                 batch_size = 64, learning_rate = 2e-5, lam = 0.9, temperature = 0.1):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = Ori_Kmeans.from_pretrained(model_path, config=self.config, cluster_centers = cluster_centers)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.lam = lam
        self.n_classes = n_classes
        self.temperature = temperature
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        

    def preparation(self, X_train, X_train_con,  y_train, X_valid, X_valid_con, y_valid, train_df):
        # create datasets
        self.train_df = train_df
        self.train_set = SCCL_CustomDataset(X_train, X_train_con,  y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = SCCL_CustomDataset(X_valid, X_valid_con,  y_valid, self.tokenizer, max_len=self.max_len)

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
        self.contrast_loss = SupConLoss(temperature=self.temperature).to(self.device)
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.alpha = 1.0

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.model.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def get_family_aug(self, data, augment_num):
        target = []
        aug_text = []
        if augment_num == 2: aug_text2 = []
        for i, label in enumerate(data["targets"]):
            target.append(int(label))

            augment_df = self.train_df[self.train_df.target_text == int(label)]
            augment_df_not = augment_df[augment_df.input_text.apply(lambda x: x not in data["text"])]
            # augment_df = augment_df[augment_df.input_text != data["text"][i]]
            if len(augment_df_not) == 0:
                print("!"*20)
                augment_df_not = augment_df[augment_df.input_text != data["text"][i]]
            augment_df = augment_df_not.sample(n=augment_num)
            agument_text = augment_df.input_text.values[0]
            aug_text.append(agument_text)
            if augment_num == 2: 
                agument_text2 = augment_df.input_text.values[1]
                aug_text2.append(agument_text2)
        aug_set = CustomDataset(aug_text, target, self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(aug_set, batch_size=self.batch_size, shuffle=False)
        for batch in train_loader:
            break
        if augment_num == 2: 
            aug_set2 = CustomDataset(aug_text2, target, self.tokenizer, max_len=self.max_len)
            train_loader2 = DataLoader(aug_set2, batch_size=self.batch_size, shuffle=False)
            for batch2 in train_loader2:
                break
            return batch, batch2
        return batch

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            input_ids1 = data["input_ids1"].to(self.device)
            attention_mask1 = data["attention_mask1"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs, hiden_state = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                outputs1, hiden_state1 = self.model(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1
                )
                features = torch.cat([F.normalize(hiden_state, dim=1).unsqueeze(1), F.normalize(hiden_state1, dim=1).unsqueeze(1)], dim=1)
                # loss = self.contrast_loss(features, targets)
                loss = self.contrast_loss(features) * 10

                cluster_output = self.get_cluster_prob(hiden_state)
                cluster_target = target_distribution(cluster_output).detach()  # pjk
                cluster_loss = self.cluster_loss((cluster_output+1e-08).log(), cluster_target)/cluster_output.shape[0]  # 做逼近
                loss += cluster_loss
                
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
        score_dict = get_MCM_score_with_new_label(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []

        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                input_ids1 = data["input_ids1"].to(self.device)
                attention_mask1 = data["attention_mask1"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, hiden_state = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                outputs1, hiden_state1 = self.model(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1
                )
                features = torch.cat([F.normalize(hiden_state, dim=1).unsqueeze(1), F.normalize(hiden_state1, dim=1).unsqueeze(1)], dim=1)
                loss = self.contrast_loss(features)

                preds = torch.argmax(outputs, dim=1).flatten()
                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                losses.append(loss.item())

        score_dict = get_MCM_score_with_new_label(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict
    
    def get_kmeans_centers(self):
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        all_labels = []
        for i, data in progress_bar:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            label = data["targets"].to(self.device)
            # outputs, bert_output = self.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask
            # )
            corpus_embeddings = self.model.get_mean_forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            all_labels += list(np.array(label.cpu()))  

            if i == 0:
                all_embeddings = corpus_embeddings.detach().cpu().numpy()
            else:
                all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().cpu().numpy()), axis=0)
        print("开始聚类")
        clustering_model = KMeans(n_clusters=self.n_classes)
        clustering_model.fit(all_embeddings)
        cluster_assignment = clustering_model.labels_

        true_labels = all_labels
        pred_labels = list(cluster_assignment)
        score_dict = get_MCM_score_with_new_label(true_labels , pred_labels )
        print(score_dict)
        return clustering_model.cluster_centers_