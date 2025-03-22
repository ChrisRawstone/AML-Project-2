import torch
from tqdm import tqdm

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    
    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (x, _) in enumerate(pbar):
                x = x.to(device)
                x = noise(x)
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}"})