import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# ---- Extracted activations ----
class NormReLU(nn.Module):
    # Implemented as described in Levinstein et al.
    def __init__(self, hidden_size, epsilon=1e-2, noise_std=0.03):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # learnable bias
        self.epsilon = epsilon
        self.noise_std = noise_std

    def forward(self, x):
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.epsilon)
        noise = torch.randn_like(x) * self.noise_std
        return F.relu(x_norm + self.bias + noise)


class HardSigmoid(nn.Module):
    # Implemented as described in Recanatesi et al. supplementary material
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.g = torch.nn.ReLU()

    def forward(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)


# ---- RNN model ----
class NextStepRNN(nn.Module):
    def __init__(self, obs_dim=144, act_dim=2, hidden_dim=500):  # Changed default act_dim from 13 to 2
        super().__init__()
        # Save dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Layers
        self.W_in = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.W_act = nn.Linear(act_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, obs_dim)  # Output = next observation
        self.beta = nn.Parameter(torch.zeros(1))
        self.norm_relu = NormReLU(hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.15)  # 15% input dropout (Table 1)
        self.hardsigmoid = HardSigmoid()

        # We can use different activation functions.
        #self.g = torch.tanh
        self.g = self.norm_relu # From Levenstein et al.
        #self.g = self.hardsigmoid # From Recanatesi et al.

        self.init_weights()

    #Init weights like in Levenstein et al.
    def init_weights(self):
        tau = 2.0
        k_in = 1.0 / np.sqrt(self.obs_dim + self.act_dim)
        k_out = 1.0 / np.sqrt(self.hidden_dim)

        init.uniform_(self.W_in.weight, -k_in, k_in)
        init.uniform_(self.W_act.weight, -k_in, k_in)
        init.uniform_(self.W_out.weight, -k_out, k_out)

        # Identity-initialized + uniform recurrent weights
        W_rec_data = torch.empty(self.hidden_dim, self.hidden_dim)
        init.uniform_(W_rec_data, -k_out, k_out)
        identity_boost = torch.eye(self.hidden_dim) * (1 - 1 / tau)
        W_rec_data += identity_boost
        self.W_rec.weight.data = W_rec_data
    '''
    #Init weights like in Recanatesi et al.
    def init_weights(self):
        # Initialize W_rec to identity matrix
        nn.init.eye_(self.W_rec.weight)

        # Initialize W_in, W_act, and W_out to normal distribution (mean=0, std=0.02)
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_act.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.02)
    '''

    def forward(self, obs_seq, act_seq, return_hidden=False):
        T, B, _ = obs_seq.size()
        h = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        y = torch.zeros(B, self.obs_dim, device=obs_seq.device)
        outputs, hiddens = [], []

        # Loop through each timestep of the agent's trajectory
        for t in range(T):
            #################### [ TO DO ] ####################
            # [1 point ] Implement the correct computation of the latent layer h
            # Hint: Look to the paper and see how the authors did this.. you can do it in one line of code
            o_in = self.W_in(obs_seq[t,:,:]) # we learn
            a_in = self.W_act(act_seq[t,:,:])
            h_in = self.W_rec(h)
            bias = self.beta
            g = self.g
            h = g(o_in + a_in + h_in + bias)
            # h =
            ###################################################
            if return_hidden:
                hiddens.append(h.detach().cpu())

            y = torch.sigmoid(self.W_out(h))
            outputs.append(y)

        outputs = torch.stack(outputs)  # (T, B, obs_dim)
        if return_hidden:
            hiddens = torch.stack(hiddens)  # (T, B, hidden_dim)
            return outputs, hiddens
        return outputs