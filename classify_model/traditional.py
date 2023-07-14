from torch import nn
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state

class TextRNNAtten(nn.Module):
    def __init__(self, config):
        super(TextRNNAtten, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        try:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        except:
            classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        # self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        output, (hidden, cell) = self.lstm(x.float())
        # output = [batch size, 12, num_directions * hidden_size]
        M = self.tanh(output)
        # M = [batch size, 12, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = output * alpha
        # print(alpha)
        # out = [batch size, 12, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, num_directions * hidden_size]
        out = F.gelu(out)
        # out = [batch size, num_directions * hidden_size]
        hidden_state = self.dense(out)
        # out = [batch size, hidden_size]
        out = self.dropout(hidden_state)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out, hidden_state
        # return out, alpha

class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch size, 12, hidden_size]
        x = self.dropout(x)
        # [batch size, text size, hidden_size]
        output, (hidden, cell) = self.lstm(x.float())
        # output = [batch size, text size, num_directions * hidden_size]
        output = torch.tanh(output)
        output = self.dropout(output)
        hidden_state = output[:, -1, :]
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output, hidden_state

class TextRNN_GRU(nn.Module):
    def __init__(self, config):
        super(TextRNN_GRU, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch size, 12, hidden_size]
        x = self.dropout(x)
        # [batch size, text size, hidden_size]
        output, hidden = self.lstm(x.float())
        # output = [batch size, text size, num_directions * hidden_size]
        output = torch.tanh(output)
        output = self.dropout(output)
        hidden_state = output[:, -1, :]
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output, hidden_state

#################################################################################################################


class LSTM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.classifier_lstm = TextRNN(config)
        self.init_weights()
        

    def forward(self, vectors):
        # Feed input to classifier to compute logits
        logits, hidden_state = self.classifier_lstm(vectors)

        return logits, hidden_state


class LSTM_Atten(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.classifier_lstm = TextRNNAtten(config)
        self.init_weights()
        

    def forward(self, vectors):
        # Feed input to classifier to compute logits
        logits, hidden_state = self.classifier_lstm(vectors)

        return logits, hidden_state

class GRU(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.classifier_lstm = TextRNN_GRU(config)
        self.init_weights()
        

    def forward(self, vectors):
        # Feed input to classifier to compute logits
        logits, hidden_state = self.classifier_lstm(vectors)

        return logits, hidden_state

class CNN(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.classifier_lstm = TextCNN(config)
        self.init_weights()
        

    def forward(self, vectors):
        # Feed input to classifier to compute logits
        logits, hidden_state = self.classifier_lstm(vectors)

        return logits, hidden_state