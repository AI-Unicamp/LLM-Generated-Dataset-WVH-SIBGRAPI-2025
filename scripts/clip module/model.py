import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel

class BlendshapeEncoder(nn.Module):
    def __init__(self, input_dim=51, latent_dim=512, dropout=0.1):
        super(BlendshapeEncoder, self).__init__()
        
        # Add a more gradual dimension reduction
        self.projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=2048,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=8
        )
        
    def forward(self, blendshape):
        x = self.projection(blendshape)
        # Add positional encoding if processing sequences
        encoded = self.encoder(x.unsqueeze(1)).squeeze(1)
        return encoded

class ClipEncoderModule(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(ClipEncoderModule, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Freeze CLIP parameters (optional)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        """
        Process a batch of texts through CLIP model to get text embeddings.
        
        Args:
            texts (list): List of strings to encode
        
        Returns:
            torch.Tensor: Batch of text embeddings
        """
        if not texts:
            return None
            
        # Handle None or empty strings in the batch
        valid_texts = [text if text else "" for text in texts]
        
        # Tokenize texts
        tokens = self.clip_tokenizer(valid_texts, 
                                     return_tensors="pt", 
                                     padding=True, 
                                     truncation=True)
        
        # Move tokens to the same device as the model
        tokens = {k: v.to(next(self.clip_model.parameters()).device) for k, v in tokens.items()}
        
        # Get text features
        with torch.no_grad():
            text_embedding = self.clip_model.get_text_features(**tokens)
            
        return text_embedding


class TextProjector(nn.Module):
    def __init__(self, input_dim=512, latent_dim=512, dropout=0.1):
        super(TextProjector, self).__init__()
        
        # Add non-linear projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, text_embedding):
        return self.projection(text_embedding)


class BlendshapeDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=51, dropout=0.1):
        super(BlendshapeDecoder, self).__init__()
        
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=2048,
                dropout=dropout
            ),
            num_layers=8
        )
        
        # Gradual dimension increase
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, output_dim),
        )

    def forward(self, latent_vector):
        decoded = self.transformer_decoder(latent_vector.unsqueeze(0))
        return self.projection(decoded.squeeze(0))