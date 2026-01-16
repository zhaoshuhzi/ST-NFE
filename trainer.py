import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MAMLTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.META_LR_OUTER)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        eeg, mri, text = [b.to(self.config.DEVICE) for b in batch]
        
        half = len(eeg) // 2
        sup_eeg, q_eeg = eeg[:half], eeg[half:]
        sup_mri, q_mri = mri[:half], mri[half:]
        sup_text, q_text = text[:half], text[half:]

        # Inner Loop (Fast Adaptation)
        logits, kl_loss = self.model(sup_eeg, sup_mri)
        sup_loss = self.criterion(logits.reshape(-1, self.config.VOCAB_SIZE), sup_text.reshape(-1))
        
        # Outer Loop (Update on Query Set)
        self.optimizer.zero_grad()
        q_logits, q_kl = self.model(q_eeg, q_mri)
        q_loss = self.criterion(q_logits.reshape(-1, self.config.VOCAB_SIZE), q_text.reshape(-1))
        
        total_loss = q_loss + 0.1 * q_kl
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def train(self, dataloader):
        self.model.train()
        pbar = tqdm(range(self.config.EPOCHS), desc="Meta-Training")
        
        for epoch in pbar:
            epoch_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
