import sys

from regex import F
sys.path.append("/mnt/qm_data/small-ase2022/classify_model")
import numpy as np
import pandas as pd
import os, torch, random
from sklearn.model_selection import train_test_split

from classify_model.ce import CE_CodeBert, load_data
from classify_model.kmeans import Kmeans
from classify_model.contras import Contras_CodeBert
from classify_model.unsuper import Unsuper_CodeBert
from classify_model.lstm import CE_LSTM, CE_GRU, CE_CNN, CE_LSTM_Atten
from classify_model.lstm_contras import Contras_CNN, Contras_GRU, Contras_LSTM, Contras_LSTM_Atten
from classify_model.lstm_contras2 import Contras2_CNN, Contras2_GRU, Contras2_LSTM, Contras2_LSTM_Atten

def set_seed(cuda_num = "1"):
    seed = 2022
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num


####   dataframe    ###########################################################################################################
def get_dataframe(filename, sub = False):
    df = pd.read_csv("./data/" + filename + ".csv")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=0)
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    test_df = eval_df.copy(deep=True) 
    if sub:
        train_df.columns = ["contrast_text", "target_text","input_text"]
        eval_df.columns = ["contrast_text", "target_text","input_text"]
        test_df.columns = ["contrast_text", "target_text","input_text"]
    else:
        train_df.columns = ["input_text", "target_text","contrast_text"]
        eval_df.columns = ["input_text", "target_text","contrast_text"]
        test_df.columns = ["input_text", "target_text","contrast_text"]
    
    return train_df, eval_df, test_df


def get_small_dataframe(filename, sub = False, num_train = 100, num_val = 500, seed = 0, nclass = 2 ):
    def get_num_df(num, df):
        df = [df.loc[df.target_text == i].sample(n=int(num / nclass), random_state=seed) for i in df.target_text.unique()]
        df = pd.concat(df, axis=0, ignore_index=True).sample(frac=1)
        return df
    df = pd.read_csv("./data/" + filename + ".csv")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=0)
    if sub:
        train_df.columns = ["contrast_text", "target_text","input_text"]
        eval_df.columns = ["contrast_text", "target_text","input_text"]
    else:
        train_df.columns = ["input_text", "target_text","contrast_text"]
        eval_df.columns = ["input_text", "target_text","contrast_text"]
    train_df = get_num_df(num_train, train_df)
    eval_df = get_num_df(num_val, eval_df)
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    test_df = eval_df.copy(deep=True) 
    return train_df, eval_df, test_df

def get_kfold_dataframe(filename, item_num = 0, sub = False):
    train_df = load_data("./data/" + filename + "_train.pkl")[item_num]
    eval_df = load_data("./data/" + filename + "_test.pkl")[item_num]
    test_df = eval_df.copy(deep=True) 
    if sub:
        train_df.columns = ["contrast_text", "target_text","input_text"]
        eval_df.columns = ["contrast_text", "target_text","input_text"]
        test_df.columns = ["contrast_text", "target_text","input_text"]
    else:
        train_df.columns = ["input_text", "target_text","contrast_text"]
        eval_df.columns = ["input_text", "target_text","contrast_text"]
        test_df.columns = ["input_text", "target_text","contrast_text"]
    
    return train_df, eval_df, test_df

####   model    ###########################################################################################################
def get_CE_CodeBert(train_df, eval_df, nclass, epochs, batch_size, filename):
    code_result_save_path = "./save/result/CE_" + filename
    code_bert_model_save_path = "./save/model/CE_" + filename + ".pt"
    classifier = CE_CodeBert(
        # model_path='microsoft/codebert-base',
        # tokenizer_path='microsoft/codebert-base',
        model_path="/root/data/qm/EL-CodeBert/codebert",
        tokenizer_path="/root/data/qm/EL-CodeBert/codebert",
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

# 仅限于从同family选取1，2个对比增加
def get_Contras_CodeBert(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.9, temperature = 0.1):
    code_result_save_path = "./save/result/Contras2_" + filename
    code_bert_model_save_path = "./save/model/Contras2_" + filename + ".pt"
    classifier = Contras_CodeBert(
        # model_path='microsoft/codebert-base',
        # tokenizer_path='microsoft/codebert-base',
        model_path="/root/data/qm/EL-CodeBert/codebert",
        tokenizer_path="/root/data/qm/EL-CodeBert/codebert",
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Unsuper_CodeBert(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.9, temperature = 0.1):
    code_result_save_path = "./save/result/Unsuper_" + filename
    code_bert_model_save_path = "./save/model/Unsuper_" + filename + ".pt"
    classifier = Unsuper_CodeBert(
        # model_path='microsoft/codebert-base',
        # tokenizer_path='microsoft/codebert-base',
        model_path="/root/data/qm/EL-CodeBert/codebert",
        tokenizer_path="/root/data/qm/EL-CodeBert/codebert",
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        X_train_con=list(train_df['contrast_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        X_valid_con=list(eval_df['contrast_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Kmeans_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.9, temperature = 0.1, cluster_centers = []):
    code_result_save_path = "./save/result/Kmeans_" + filename
    code_bert_model_save_path = "./save/model/Kmeans_" + filename + ".pt"
    classifier = Kmeans(
        model_path="/root/data/qm/EL-CodeBert/codebert",
        tokenizer_path="/root/data/qm/EL-CodeBert/codebert",
        cluster_centers=cluster_centers, 
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        X_train_con=list(train_df['contrast_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        X_valid_con=list(eval_df['contrast_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, filename):
    code_result_save_path = "./save/result/LSTM_" + filename
    code_bert_model_save_path = "./save/model/LSTM_" + filename + ".pt"
    classifier = CE_LSTM(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name


def get_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, filename):
    code_result_save_path = "./save/result/LSTM_Atten_" + filename
    code_bert_model_save_path = "./save/model/LSTM_Atten_" + filename + ".pt"
    classifier = CE_LSTM_Atten(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, filename):
    code_result_save_path = "./save/result/GRU_" + filename
    code_bert_model_save_path = "./save/model/GRU_" + filename + ".pt"
    classifier = CE_GRU(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, filename):
    code_result_save_path = "./save/result/CNN_" + filename
    code_bert_model_save_path = "./save/model/CNN_" + filename + ".pt"
    classifier = CE_CNN(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name


def get_Contras_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras_LSTM_" + filename
    code_bert_model_save_path = "./save/model/Contras_LSTM_" + filename + ".pt"
    classifier = Contras_LSTM(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Contras2_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras2_LSTM_" + filename
    code_bert_model_save_path = "./save/model/Contras2_LSTM_" + filename + ".pt"
    classifier = Contras2_LSTM(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name


def get_Contras_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras_LSTM_Atten_" + filename
    code_bert_model_save_path = "./save/model/Contras_LSTM_Atten_" + filename + ".pt"
    classifier = Contras_LSTM_Atten(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Contras2_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras2_LSTM_Atten_" + filename
    code_bert_model_save_path = "./save/model/Contras2_LSTM_Atten_" + filename + ".pt"
    classifier = Contras2_LSTM_Atten(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Contras_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras_GRU_" + filename
    code_bert_model_save_path = "./save/model/Contras_GRU_" + filename + ".pt"
    classifier = Contras_GRU(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Contras2_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras2_GRU_" + filename
    code_bert_model_save_path = "./save/model/Contras2_GRU_" + filename + ".pt"
    classifier = Contras2_GRU(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

def get_Contras_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras_CNN_" + filename
    code_bert_model_save_path = "./save/model/Contras_CNN_" + filename + ".pt"
    classifier = Contras_CNN(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name


def get_Contras2_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, filename, lam = 0.3, temperature = 0.1):
    code_result_save_path = "./save/result/Contras2_CNN_" + filename
    code_bert_model_save_path = "./save/model/Contras2_CNN_" + filename + ".pt"
    classifier = Contras2_CNN(
        model_path='microsoft/codebert-base',
        max_len=256,
        n_classes=nclass,
        epochs=epochs,
        model_save_path=code_bert_model_save_path,
        result_save_path = code_result_save_path,
        batch_size=batch_size,
        learning_rate=2e-5,
        lam = lam,
        temperature = temperature
    )
    classifier.preparation(
        X_train=list(train_df['input_text']),
        y_train=list(train_df['target_text']),
        X_valid=list(eval_df['input_text']),
        y_valid=list(eval_df['target_text']),
        train_df = train_df
    )
    pretrained_model_name = code_bert_model_save_path
    return classifier, pretrained_model_name

####   get_entry    ###########################################################################################################

def get_ce(filename, nclass, epochs = 20,batch_size = 32, Load_Pretrained = False, sub = False, \
                    small = False, num_train = 100 , num_val = 500, seed = 0):
    train_df, eval_df, test_df = get_dataframe(filename, sub = sub) if small == False \
        else get_small_dataframe(filename, sub = sub, num_train = num_train , num_val = num_val, seed = seed, nclass = nclass)
    save_filename = filename + "_bat" + str(batch_size) if sub == False \
        else filename + "_sub_bat" + str(batch_size) 
    if small: save_filename += "_seed" + str(seed) + "_" + str(num_train) + "_" + str(num_val)
    classifier, pretrained_model_name = get_CE_CodeBert(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_ce over!")

def get_contras(filename, nclass, batch_size = 32, epochs = 20,  Load_Pretrained = False, sub = False, \
                    small = False, num_train = 100 , num_val = 500, seed = 0, lam = 0.9, temperature = 0.1):
    train_df, eval_df, test_df = get_dataframe(filename, sub = sub) if small == False \
        else get_small_dataframe(filename, sub = sub, num_train = num_train , num_val = num_val, seed = seed, nclass = nclass)
    save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
        else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
    if small: save_filename += "_seed" + str(seed) + "_" + str(num_train) + "_" + str(num_val)
    classifier, pretrained_model_name = get_Contras_CodeBert(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_contras over!")

def get_unsuper(filename, nclass, batch_size = 32, epochs = 20,  Load_Pretrained = False, sub = False, \
                    small = False, num_train = 100 , num_val = 500, seed = 0):
    lam = 0.9
    temperature = 0.1
    train_df, eval_df, test_df = get_dataframe(filename, sub = sub) if small == False \
        else get_small_dataframe(filename, sub = sub, num_train = num_train , num_val = num_val, seed = seed, nclass = nclass)
    save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
        else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
    if small: save_filename += "_seed" + str(seed) + "_" + str(num_train) + "_" + str(num_val)
    classifier, pretrained_model_name = get_Unsuper_CodeBert(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_unsuper over!")

def get_kmeans(filename, nclass, batch_size = 32, epochs = 20,  Load_Pretrained = False, sub = False, \
                    small = False, num_train = 100 , num_val = 500, seed = 0):
    lam = 0.9
    temperature = 0.1
    train_df, eval_df, test_df = get_dataframe(filename, sub = sub) if small == False \
        else get_small_dataframe(filename, sub = sub, num_train = num_train , num_val = num_val, seed = seed, nclass = nclass)
    save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
        else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
    if small: save_filename += "_seed" + str(seed) + "_" + str(num_train) + "_" + str(num_val)
    classifier, pretrained_model_name = get_Kmeans_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    cluster_centers = classifier.get_kmeans_centers()

    classifier, pretrained_model_name = get_Kmeans_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature, cluster_centers)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train()
    print("train_kmeans over!")

def get_kfold_ce(filename, nclass, epochs = 20, batch_size = 32, Load_Pretrained = False, sub = False, item_num = 0):
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename
    save_filename = filename + "_bat" + str(batch_size) if sub == False \
        else filename + "_sub_bat" + str(batch_size) 
    classifier, pretrained_model_name = get_CE_CodeBert(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_kfold_ce over!")

def get_kfold_contras(filename, nclass, batch_size = 32, epochs = 20,  Load_Pretrained = False, sub = False, \
                    lam = 0.9, temperature = 0.1, item_num = 0):
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename
    save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
        else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
    classifier, pretrained_model_name = get_Contras_CodeBert(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_kfold_contras over!")

def get_traditional_lstm(filename, nclass, epochs = 20,batch_size = 32, Load_Pretrained = False, sub = True, item_num = 0,\
                    lam = 0.3, temperature = 0.1, contras = 0):
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename
    
    if contras == 0:
        save_filename = filename + "_bat" + str(batch_size) if sub == False \
            else filename + "2_sub_bat" + str(batch_size) 
        classifier, pretrained_model_name = get_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    elif contras == 1 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    elif contras == 2 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras2_LSTM_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    

    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_ce over!")

def get_traditional_lstm_atten(filename, nclass, epochs = 20,batch_size = 32, Load_Pretrained = False, sub = True, item_num = 0,\
                    lam = 0.3, temperature = 0.1, contras = 0):
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename
    
    if contras == 0:
        save_filename = filename + "_bat" + str(batch_size) if sub == False \
            else filename + "2_sub_bat" + str(batch_size) 
        classifier, pretrained_model_name = get_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    elif contras == 1 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    elif contras == 2 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras2_LSTM_Atten_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    

    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_ce over!")

def get_traditional_gru(filename, nclass, epochs = 20,batch_size = 32, Load_Pretrained = False, sub = True, item_num = 0,\
                    lam = 0.3, temperature = 0.1, contras = 0):
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename

    if contras == 0:
        save_filename = filename + "_bat" + str(batch_size) if sub == False \
            else filename + "2_sub_bat" + str(batch_size) 
        classifier, pretrained_model_name = get_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    elif contras == 1 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    elif contras == 2:
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras2_GRU_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)


    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_ce over!")

def get_traditional_cnn(filename, nclass, epochs = 20,batch_size = 32, Load_Pretrained = False, sub = True, item_num = 0,\
                    lam = 0.3, temperature = 0.1, contras = 0):   
    train_df, eval_df, test_df = get_kfold_dataframe(filename, sub = sub, item_num = item_num) 
    filename = "Kfold_" + str(item_num) + "_" + filename

    if contras == 0:
        save_filename = filename + "2_bat" + str(batch_size) if sub == False \
            else filename + "2_sub_bat" + str(batch_size) 
        classifier, pretrained_model_name = get_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename)
    elif contras == 1 :
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)
    elif contras == 2:
        save_filename = filename + "_bat" + str(batch_size) +"_lam" + str(lam) +"_tem" + str(temperature) if sub == False\
            else filename + "_sub_bat" + str(batch_size)  +"_lam" + str(lam) +"_tem" + str(temperature)
        classifier, pretrained_model_name = get_Contras2_CNN_Model(train_df, eval_df, nclass, epochs, batch_size, save_filename, lam , temperature)

    if Load_Pretrained == True and os.path.exists(pretrained_model_name):
        classifier.model.load_state_dict(torch.load(pretrained_model_name))
    classifier.train(Load_Pretrained)
    print("train_ce over!")




####   main_entry    ###########################################################################################################

def main_small(epoch = 20):
    cuda_num = "1"
    set_seed(cuda_num)
    Load_Pretrained = True
    
    seed = 0
    small = True
    num_train = 100
    num_val = 500
    for seed in range(10):
        for num_train in [20, 100, 50, 1000]:
            batch_size = 32 if num_train != 20 else 16
            filename = "real_nr_2"
            get_ce(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, small=small, num_train=num_train, num_val=num_val, seed=seed)
            get_contras(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, small=small, num_train=num_train, num_val=num_val, seed=seed)
            # get_ce(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True, small=small, num_train=num_train, num_val=num_val, seed=seed)
            # get_contras(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True, small=small, num_train=num_train, num_val=num_val, seed=seed)

            filename = "sard_nr_2"
            get_ce(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, small=small, num_train=num_train, num_val=num_val, seed=seed)
            get_contras(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, small=small, num_train=num_train, num_val=num_val, seed=seed)
            # get_ce(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True, small=small, num_train=num_train, num_val=num_val, seed=seed)
            # get_contras(filename, 2, batch_size = batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True, small=small, num_train=num_train, num_val=num_val, seed=seed)

def main(epoch = 5):
    Load_Pretrained = True

    # filename = "realdata_2"
    # get_ce(filename, 2, Load_Pretrained = True, epochs=30)
    # get_contras(filename, 2, Load_Pretrained = True, epochs=50)
    
    # filename = "sard_nr_2"
    # get_ce(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch)
    # get_contras(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch)
    # get_ce(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True)
    # get_contras(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True)

    # filename = "sard_2"

    # filename = "real_nr_2"
    # get_ce(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch)
    # get_contras(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch)
    # get_ce(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True)
    # get_contras(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch, sub = True)

    filename = "mvd_nr_40"
    get_ce(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch)
    get_contras(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch)


def main_unsuper(epoch = 20):
    Load_Pretrained = False

    # filename = "sard_nr_2"
    filename = "real_nr_2"
    get_unsuper(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch)


def main_kmeans(epoch = 20):
    Load_Pretrained = False
    filename = "sard_nr_2"
    get_kmeans(filename, 2, Load_Pretrained = Load_Pretrained, epochs=epoch,small=True, num_train=20, num_val=100, seed=0)


def main_kfold(epoch = 5):
    Load_Pretrained = True
    for item_num in range(10):
        # 10组交叉验证100-10-10下采样样本（方法不合理，废弃）
        # filename = "mvd_nr_40"
        # get_kfold_ce(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num)
        # get_kfold_contras(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num)

        # # 10组交叉验证100-10-10下采样样本
        # filename = "mvd_nr_40_nsr"
        # get_kfold_ce(filename, 40, Load_Pretrained = Load_Pretrained, epochs=40, item_num = item_num)
        # for lam in [0.1, 0.3, 0.5]:    
        #     for temperature in [0.1, 0.3]:
        #         get_kfold_contras(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num,\
        #             lam = lam, temperature = temperature)
        
        # 10组交叉验证全集样本
        epoch = 30
        filename = "mvd_allkfold_40"
        get_kfold_ce(filename, 40, Load_Pretrained = Load_Pretrained, epochs=20, item_num = item_num)
        for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:  # [0.1, 0.3, 0.5, 0.7, 0.9]
            for temperature in [0.1, 0.3, 0.5]:  # [0.1,0.3,0.5, 0.7]
                get_kfold_contras(filename, 40, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num,\
                    lam = lam, temperature = temperature)


def main_lstm(contras = True, batch_size = 256, epoch = 100, sub = True, filename = "mvd_down_40", max_item = 5):
    Load_Pretrained = True
    # for item_num in range(max_item):
    for item_num in [1,3,4,5,6,8]:
        
        # filename = "mvd_allkfold_40"
        get_traditional_lstm(filename, 40, sub = sub, batch_size=batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num, contras = contras)

def main_lstm_atten(contras = True, batch_size = 256, epoch = 100, sub = True, filename = "mvd_down_40", max_item = 5):
    Load_Pretrained = True
    for item_num in range(max_item):
        # filename = "mvd_allkfold_40"
        get_traditional_lstm_atten(filename, 40, sub = sub, batch_size=batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num, contras = contras)


def main_cnn(contras = True, batch_size = 256, epoch = 100, sub = True, filename = "mvd_down_40", max_item = 5):
    Load_Pretrained = True
    for item_num in range(max_item):
        # filename = "mvd_allkfold_40"
        get_traditional_cnn(filename, 40, sub = sub, batch_size=batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num, contras = contras)
        # for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:  # [0.1, 0.3, 0.5, 0.7, 0.9]
        #     for temperature in [0.1, 0.3, 0.5, 0.7]:  # [0.1,0.3,0.5, 0.7]
        #         get_traditional_cnn(filename, 40, sub = sub, batch_size=batch_size,lam = lam, temperature = temperature, \
        #             Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num, contras = contras)


def main_gru(contras = True, batch_size = 256, epoch = 100, sub = True, filename = "mvd_down_40", max_item = 5):
    Load_Pretrained = True
    for item_num in range(max_item):
        # filename = "mvd_allkfold_40"
        get_traditional_gru(filename, 40, sub = sub, batch_size=batch_size, Load_Pretrained = Load_Pretrained, epochs=epoch, item_num = item_num, contras = contras)

####   start    ###########################################################################################################

if __name__ == "__main__":
    cuda_num = "1"
    set_seed(cuda_num)
    # main_kfold(60)
    # main_small(50)
    main(30)
    
    # main_unsuper()
    # main_kmeans()

# traditional
    contras = 1
    epoch = 500
    sub = True
    batch_size = 32
    max_item = 10
    filename = "mvd_down_40"
    
    # main_cnn(contras, batch_size, sub = sub, epoch= epoch, filename = filename, max_item = max_item)
    # main_gru(contras, batch_size, sub = sub, epoch= epoch, filename = filename, max_item = max_item)
    main_lstm(contras, batch_size, sub = sub, epoch= epoch, filename = filename, max_item = max_item)
    # main_lstm_atten(contras, batch_size, sub = sub, epoch= epoch, filename = filename, max_item = max_item)