"""All three sequence baselines share the same observation and missingness encoding."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def encode(raw):
    valid = torch.isfinite(raw)
    x = torch.where(valid, raw, 0.0).clamp(0, 1)
    channels = (
        x,
        valid,
        x == 0,
        x == 1,
        -x.clamp_min(1e-6).log() / math.log(1e6),
        -(1 - x).clamp_min(1e-6).log() / math.log(1e6),
    )
    return (torch.stack(channels, -1) * valid.unsqueeze(-1)).flatten(-2)


class Residual(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))

    def forward(self, x):
        return x + super().forward(x)


class HeaderNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.embed = nn.Sequential(nn.Linear(619, 32), nn.SiLU())
        pooled = 64
        if name == "header-mil":
            self.instance = Residual()
            self.attention = nn.Linear(32, 1)
            pooled = 96
        elif name == "header-tcn":
            self.convs = nn.ModuleList(
                nn.Conv1d(32, 32, 3, padding=d, dilation=d) for d in (1, 2, 4)
            )
        elif name == "header-gru":
            self.gru = nn.GRUCell(32, 32)
            pooled = 32
        else:
            raise ValueError(name)
        self.head = nn.Sequential(
            nn.Linear(pooled + 350, 32), nn.SiLU(), Residual(), nn.Linear(32, 1)
        )
        self.register_buffer("position", torch.linspace(-1, 1, 16).reshape(1, -1, 1))

    def forward(self, bins, style):
        valid = torch.isfinite(bins)
        mask = valid.any(-1)
        m = mask.unsqueeze(-1)
        h = (
            self.embed(
                torch.cat([encode(bins), self.position.expand(len(bins), -1, -1)], -1)
            )
            * m
        )
        if self.name == "header-gru":
            pooled = h.new_zeros(len(h), 32)
            for i in range(16):
                pooled = torch.where(m[:, i], self.gru(h[:, i], pooled), pooled)
        else:
            if self.name == "header-mil":
                h = self.instance(h) * m
            else:
                for conv in self.convs:
                    h = (h + F.silu(conv(h.transpose(1, 2)).transpose(1, 2))) * m
            mx = h.masked_fill(~m, -torch.inf).amax(1)
            mean = h.sum(1) / m.sum(1).clamp_min(1)
            parts = [mx, mean]
            if self.name == "header-mil":
                att = self.attention(h).masked_fill(~m, -torch.inf).softmax(1)
                parts.insert(0, (att * h).sum(1))
            pooled = torch.cat(parts, -1)
        return self.head(
            torch.cat(
                [
                    pooled,
                    ((bins > 0) & valid).any(1),
                    ((bins < 1) & valid).any(1),
                    encode(style),
                ],
                -1,
            )
        ).squeeze(-1)
