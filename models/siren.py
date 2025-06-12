import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_feat, lat_feat, out_feat, bias=True, is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_feat
        self.out_features = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.linear_lats = nn.Linear(lat_feat, out_feat * 2, bias=bias) if lat_feat > 0 else None
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                            np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        """
        input: (coords, latent_vec)
        """
        intermed = self.linear(input[0])
        if self.linear_lats is not None:
            lats = self.linear_lats(input[1])
            out = torch.sin((self.omega * intermed * lats[..., :self.out_features]) + lats[..., self.out_features:])
        else:
            out = torch.sin(self.omega * intermed)
        return out, input[1]


class Siren(nn.Module):
    def __init__(self, in_size, lat_size, out_size, hidden_size, num_layers, f_om, h_om,
                 outermost_linear, modulated_layers):
        super().__init__()
        l_in_mod = 0 in modulated_layers
        self.net = [SineLayer(in_size, lat_size * l_in_mod, hidden_size, is_first=True, omega=f_om)]
        self.hidden_size = hidden_size
        for i in range(num_layers):
            l_in_mod = (i+1) in modulated_layers
            self.net.append(SineLayer(hidden_size, lat_size * l_in_mod, hidden_size, is_first=False, omega=h_om))

        if outermost_linear:
            self.final_linear = nn.Linear(hidden_size+lat_size, out_size, bias=True)
            with torch.no_grad():
                self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / h_om,
                                                np.sqrt(6 / hidden_size) / h_om)
        else:
            self.final_linear = SineLayer(hidden_size, 0, out_size, is_first=False, omega=h_om)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return self.final_linear(torch.cat([x[0], x[1]], dim=-1))
