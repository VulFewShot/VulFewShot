import os
import torch, pickle
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup


from classify_model.dataset import CustomDataset, TraditionalDataset
from classify_model.score import get_MCM_score
from classify_model.ori import Ori_CodeBert
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

class CE_CodeBert:
    def __init__(self, model_path, tokenizer_path, max_len=256, n_classes=2, epochs=15,
                 model_save_path='./save/codebert.pt',result_save_path = './save/result/codebert.pt', batch_size = 64, learning_rate = 2e-5):
        self.config = RobertaConfig.from_pretrained(model_path, num_labels = n_classes)
        self.model = Ori_CodeBert.from_pretrained(model_path, config=self.config)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
    
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
    
    def get_traditional_family_aug(self, data, augment_num, embeddings):
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
        aug_set = TraditionalDataset(aug_text, target, self.max_len, embeddings)
        train_loader = DataLoader(aug_set, batch_size=self.batch_size, shuffle=False)
        for batch in train_loader:
            break
        if augment_num == 2: 
            aug_set2 = TraditionalDataset(aug_text2, target, self.max_len, embeddings)
            train_loader2 = DataLoader(aug_set2, batch_size=self.batch_size, shuffle=False)
            for batch2 in train_loader2:
                break
            return batch, batch2
        return batch
        

    def preparation(self, X_train,  y_train, X_valid, y_valid, train_df):
        # create datasets
        self.train_df = train_df
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer, max_len=self.max_len)

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
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
    
    def train(self, Load_Pretrained=False):
        best_accuracy = 0
        learning_record_dict = {}
        start = 0
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        if Load_Pretrained == True and os.path.exists(self.result_save_path + ".result"):
            learning_record_dict = load_data(self.result_save_path + ".result")
            start = len(learning_record_dict)
            for i in learning_record_dict:
                train_table.add_row(["tra", str(i+1), format(learning_record_dict[i]["train_loss"], '.4f')] + \
                    [learning_record_dict[i]["train_score"][j] for j in learning_record_dict[i]["train_score"] if j != "MCM"])
                test_table.add_row(["val", str(i+1), format(learning_record_dict[i]["val_loss"], '.4f')] + \
                    [learning_record_dict[i]["val_score"][j] for j in learning_record_dict[i]["val_score"] if j != "MCM"])
        for epoch in range(start, self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            
            if float(val_score["M_f1"]) > best_accuracy:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = float(val_score["M_f1"])
            
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                "train_score": train_score, "val_score": val_score}
            sava_data(self.result_save_path + ".result", learning_record_dict)
            print("\n")

        if self.epochs == 0:
            torch.save(self.model.state_dict(), self.model_save_path )
        else:
            torch.save(self.model.state_dict(), self.model_save_path + "-every")
        self.model.load_state_dict(torch.load(self.model_save_path))
