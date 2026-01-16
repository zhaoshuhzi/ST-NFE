import torch
import torch.nn as nn
import torch.autograd as autograd

# --- 1. NPI Module (Temporal Stream) ---
class SurrogateModel(nn.Module):
    """NPI Surrogate Model: Simulates Brain Dynamics f(x_t) -> x_{t+1}"""
    def __init__(self, channels):
        super().__init__()
        self.lstm = nn.LSTM(channels, 128, batch_first=True)
        self.head = nn.Linear(128, channels)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)

class NPITemporalEncoder(nn.Module):
    """NPI Encoder: Captures Causal Flow via Virtual Perturbation (Gradient-based approximation)"""
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.surrogate = SurrogateModel(channels)
        self.encoder = nn.GRU(channels, latent_dim, batch_first=True)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # A. Virtual Perturbation (Using Gradient as Sensitivity Measure)
        x_in = x.clone().requires_grad_(True)
        pred = self.surrogate(x_in)
        # Calculate gradient of output sum w.r.t input (Effective Connectivity proxy)
        grad_sum = autograd.grad(outputs=pred.sum(), inputs=x_in, create_graph=True)[0]
        
        # B. Sequence Encoding
        _, hn = self.encoder(grad_sum)
        hn = hn[-1]
        
        return self.fc_mu(hn), self.fc_logvar(hn)

# --- 2. Spatial DNN (Spatial Stream) ---
class SpatialDNN(nn.Module):
    """Reconstructs Source Space from MRI"""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten()
        )
        flat_dim = 32 * 8 * 8 * 8
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        feat = self.net(x)
        return self.fc_mu(feat), self.fc_logvar(feat)

# --- 3. ST-NFE Main Model ---
class ST_NFE(nn.Module):
    """Spatiotemporal Neural Field Encoder: Fusion of NPI (Time) and DNN (Space) with Semantic Alignment"""
    def __init__(self, config):
        super().__init__()
        self.npi_stream = NPITemporalEncoder(config.EEG_CHANNELS, config.LATENT_DIM)
        self.spatial_stream = SpatialDNN(config.LATENT_DIM)
        
        self.fusion = nn.Linear(config.LATENT_DIM * 2, config.LATENT_DIM)
        
        self.decoder = nn.LSTM(config.LATENT_DIM, 256, batch_first=True)
        self.head = nn.Linear(256, config.VOCAB_SIZE)
        self.seq_len = config.SEQ_LEN

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, eeg, mri):
        # 1. Temporal Stream (NPI) -> Distribution Q
        mu_t, logvar_t = self.npi_stream(eeg)
        z_t = self.reparameterize(mu_t, logvar_t)
        
        # 2. Spatial Stream (DNN) -> Distribution P
        mu_s, logvar_s = self.spatial_stream(mri)
        z_s = self.reparameterize(mu_s, logvar_s)
        
        # 3. Posterior Learning (KL Loss: Align Time to Space)
        kl_loss = -0.5 * torch.sum(1 + logvar_t - logvar_s - (mu_t - mu_s).pow(2) / logvar_s.exp(), dim=1).mean()
        
        # 4. Neural Field Construction
        w_dw = self.fusion(torch.cat([z_t, z_s], dim=1))
        
        # 5. Semantic Decoding
        decoder_in = w_dw.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(decoder_in)
        logits = self.head(out)
        
        return logits, kl_loss
