import torch
from torch import nn
from torch.nn import Sequential
from hw_asr.base import BaseModel
import torch.nn.functional as F



class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):
        return {"logits": self.net(spectrogram.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, bias=True, dropout=0.1, bidirectional=True, batch_first=True)

    def forward(self, inputs):
        outputs = F.relu(self.bn(inputs.transpose(1, 2)))
        outputs, _ = self.rnn(outputs.transpose(1, 2))
        return outputs


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, num_rnn_layers=5, rnn_hidden_size=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        self.layers = nn.ModuleList()
        for i in range(num_rnn_layers):
            self.layers.append(BlockRNN(input_size= 8 * n_feats if i == 0 else 2*rnn_hidden_size, hidden_size=rnn_hidden_size))
        self.fc = nn.Sequential(nn.LayerNorm(2*rnn_hidden_size), nn.Linear(2*rnn_hidden_size, n_class, bias=False),)

    def forward(self, spectrogram, *args, **kwargs):
        outputs = self.conv(spectrogram.unsqueeze(1))
        input_size, channels_siz, hide, seq_size = outputs.size()
        outputs = outputs.view(input_size, channels_siz * hide, seq_size).permute(2, 0, 1)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = outputs.permute(1, 0, 2)
        return self.fc(outputs)

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2