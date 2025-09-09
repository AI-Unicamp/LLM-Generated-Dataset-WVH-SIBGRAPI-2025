import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predicted, target):
        return self.criterion(predicted, target)

class AlignLoss(nn.Module):
    def __init__(self, device=None):
        super(AlignLoss, self).__init__()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.device = device

    def forward(self, text_projected, blend_latent, label):
        # Ensure that all tensors are on the same device
        text_projected = text_projected.to(self.device)
        blend_latent = blend_latent.to(self.device)
        label = label.to(self.device)
        
        # Check the shape for debugging
        # print(text_projected.shape)
        # print(blend_latent.shape)
        
        # CosineEmbeddingLoss expects inputs: (x1, x2, target)
        # target: 1 if similar, -1 if dissimilar
        loss = self.cos_loss(text_projected, blend_latent, label)
        return loss