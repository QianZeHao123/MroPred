import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnModel(nn.Module):

    def __init__(
        self,
        rnn_type: str,
        input_size,
        rnn_output_size,
        output_size,
        bidirectional: bool = False,
        num_layers: int = 3,
        pooling_method: str = "attention",  # value =  None, 'max', 'avg', 'attention', 'multihead_attention'
        num_heads=4,
        use_last_hidden: bool = True,
    ):
        super(RnnModel, self).__init__()
        # bidirectional = False
        num_directions = 2 if bidirectional else 1
        self.embed_dim = rnn_output_size * num_directions
        self.pooling_method = pooling_method
        self.use_last_hidden = use_last_hidden

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_output_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=rnn_output_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        if pooling_method == "multihead_attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
            )
        elif pooling_method == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=1, batch_first=True
            )
        elif pooling_method in ["max", "avg"]:
            self.pool = getattr(nn, f"Adaptive{pooling_method.capitalize()}Pool1d")(1)
        elif pooling_method is None:
            pass
        else:
            raise ValueError(f"Unsupported pooling_method: {pooling_method}")
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size),
        )

    def forward(self, x, length: torch.Tensor):
        # x (batch_size, seq_len, input_size)
        # use pack_padded_sequence and pad_packed_sequence to deal with different length of x input
        packed_x = pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # pooled = self.pool(rnn_out.transpose(1, 2)).squeeze(2)
        # attended = self.attention(rnn_out)
        if self.pooling_method is None:
            if self.use_last_hidden:
                idx = (length - 1).to(x.device)
                batch_indices = torch.arange(x.size(0), device=x.device)
                pooled = rnn_out[batch_indices, idx]  # (B, H)
            else:
                # use the average to of all outputs
                pooled = rnn_out.mean(dim=1)  # (B, H)
        elif self.pooling_method in ["max", "avg"]:
            pooled = self.pool(rnn_out.transpose(1, 2)).squeeze(2)
        elif self.pooling_method in ["attention", "multihead_attention"]:
            # if self.pooling_method == "multihead_attention":
            #     attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
            #     pooled = attn_out.mean(dim=1)
            # else:
            #     pooled = self.attn(rnn_out)
            attn_output, _ = self.attn(rnn_out, rnn_out, rnn_out)
            pooled = attn_output.mean(dim=1)
        else:
            raise ValueError("Invalid pooling_method")

        model_out = self.fc(pooled)
        return model_out
