# Trimmed version of the original NN.py.
# IntegratedModel is kept, but ONLY the GRU and LSTM paths remain.
# The attention and SSM (Mamba) branches were removed, so there is no
# dependency on `attention` or `mamba_ssm` anymore and the file loads cleanly.

import torch
import torch.nn as nn


class IntegratedModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        GRU_blocks: int,
        LSTM_blocks: int,
        Atten_blocks: int,
        SSM_blocks: int,
        d_intermediate: int,
        d_hidden: int,
        d_outfeature: int,
        dim_atten: int,
        n_heads: int,
        dim_seq: int,
        **kwargs,                       # absorbs the now-unused ssm/attn kwargs
    ) -> None:
        super().__init__()

        # This trimmed model only supports GRU and LSTM.
        assert Atten_blocks == 0 and SSM_blocks == 0, (
            "This trimmed NN.py supports only GRU and LSTM "
            "(set Atten_blocks=0 and SSM_blocks=0)."
        )

        # replaces the embedding layer: maps the input features to d_hidden
        self.proj_in = nn.Linear(dim_in, d_hidden)
        # collapses the time dimension (dim_seq steps) into one
        self.proj_out1 = nn.Conv1d(dim_seq, 1, 1)
        # maps the hidden vector to the forecast horizon
        self.proj_out2 = nn.Linear(d_hidden, d_outfeature)

        self.GRU_blocks = GRU_blocks
        if self.GRU_blocks > 0:
            self.gru = nn.GRU(d_hidden, d_hidden, GRU_blocks,
                              batch_first=True, dropout=0)

        self.LSTM_blocks = LSTM_blocks
        if self.LSTM_blocks > 0:
            self.lstm = nn.LSTM(d_hidden, d_hidden, LSTM_blocks,
                                batch_first=True, dropout=0)

    def forward(self, input_ids):
        hidden_states = self.proj_in(input_ids)        # (B, L, d_hidden)

        if self.GRU_blocks > 0:
            hidden_states, _ = self.gru(hidden_states)
        if self.LSTM_blocks > 0:
            hidden_states, _ = self.lstm(hidden_states)

        hidden_states = self.proj_out1(hidden_states)  # (B, 1, d_hidden)
        # squeeze only the time axis (squeeze() alone would also drop batch=1)
        hidden_states = self.proj_out2(hidden_states.squeeze(1))  # (B, horizon)
        return hidden_states
