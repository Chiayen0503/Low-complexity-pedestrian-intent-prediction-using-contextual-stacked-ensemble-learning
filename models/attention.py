import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnType:
    GRU = 1
    LSTM = 2

class AttentionModel:
    NONE = 0
    DOT = 1
    GENERAL = 2
    CONCAT = 3

class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))

class Attention(nn.Module):
    def __init__(self, device, method, hidden_size):
        super(Attention, self).__init__()
        self.device = device

        self.method = method
        self.hidden_size = hidden_size

        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.method == AttentionModel.GENERAL:
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == AttentionModel.CONCAT:
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = torch.FloatTensor(1, hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        batch_size, seq_len, _ = rnn_outputs.shape
        if self.method == AttentionModel.DOT:
            attn_weights = torch.bmm(rnn_outputs, final_hidden_state.unsqueeze(2))
        elif self.method == AttentionModel.GENERAL:
            attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
            attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        #elif self.method == AttentionModel.CONCAT:
        #    NOT IMPLEMENTED
        else:
            raise Exception("[Error] Unknown AttentionModel.")

        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

        return attn_hidden, attn_weights