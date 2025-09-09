import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from loss import MSELoss, AlignLoss
from sparse_losses import SparseAwareMSELoss
import os

class Trainer:
    def __init__(self, 
                 encoder, 
                 decoder, 
                 projector, 
                 clip_encoder,
                 dataset, 
                 batch_size=None, 
                 learning_rate=None, 
                 epochs=None, 
                 device=None,
                 loss_weights=None,
                 val_split=0.2):
        self.encoder = encoder
        self.decoder = decoder
        self.clip_encoder = clip_encoder
        self.projector = projector

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.custom_collate)
        self.epochs = epochs
        self.device = device

        # Split dataset into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.custom_collate)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.custom_collate)

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.projector.to(self.device)
        self.clip_encoder.to(self.device)
        
        self.optimizer = Adam(
            list(encoder.parameters()) + 
            list(decoder.parameters()) + 
            list(projector.parameters()), 
            lr=learning_rate
        )

        self.criterion_1 = MSELoss().to(self.device)
        self.criterion_2 = AlignLoss().to(self.device)
        
        self.loss_weights = loss_weights or {"reconstruction": 1, "alignment": 10, "crossmodal": 10}

    def custom_collate(self, batch):
        """Custom collate function to handle multiple augmentations per sample."""
        all_texts = []
        all_blendshapes = []

        for text_list, blendshape in batch:
            all_texts.extend(text_list)  # Flatten the text lists
            all_blendshapes.extend([blendshape] * len(text_list))  # Duplicate blendshapes per augmentation
        
        return all_texts, torch.stack(all_blendshapes)
    
    def save_model(self, model, file_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save only the encoder state dictionary
        torch.save(model.state_dict(), file_path)
        print(f"{model} saved to {file_path}")

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.projector.train()

        for epoch in range(self.epochs):
            train_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, (text_list, blendshapes) in enumerate(progress_bar):
                blendshapes = blendshapes.to(self.device)

                #print(f'text_list: {text_list}')

                # Process text through CLIP with augmented data
                text_clip = self.clip_encoder(text_list)
                
                # Project CLIP embeddings
                text_proj = self.projector(text_clip)

                # Encode blendshapes
                latent_vector = self.encoder(blendshapes)

                # Decode
                bs_from_bs = self.decoder(latent_vector)
                
                # Calculate losses
                label = torch.ones(latent_vector.shape[0], device=self.device)

                loss_1 = self.criterion_1(bs_from_bs, blendshapes)
                loss_2 = self.criterion_2(text_proj, latent_vector, label)
                loss_3 = self.criterion_1(self.decoder(text_proj), blendshapes)

                # Final loss
                loss = (self.loss_weights['reconstruction'] * loss_1 + 
                        self.loss_weights['alignment'] * loss_2 + 
                        self.loss_weights['crossmodal'] * loss_3)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})

            avg_train_loss = train_loss / len(self.train_loader)

            # Validation phase
            self.encoder.eval()
            self.decoder.eval()
            self.projector.eval()
            
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (text_list, blendshapes) in enumerate(self.val_loader):
                    blendshapes = blendshapes.to(self.device)

                    # Process text through CLIP with augmented data
                    text_clip = self.clip_encoder(text_list)
                    
                    # Project CLIP embeddings
                    text_proj = self.projector(text_clip)

                    # Encode blendshapes
                    latent_vector = self.encoder(blendshapes)

                    # Decode
                    bs_from_bs = self.decoder(latent_vector)
                    
                    # Calculate losses
                    label = torch.ones(latent_vector.shape[0], device=self.device)

                    loss_1 = self.criterion_1(bs_from_bs, blendshapes)
                    loss_2 = self.criterion_2(text_proj, latent_vector, label)
                    loss_3 = self.criterion_1(self.decoder(text_proj), blendshapes)

                    # Final loss
                    loss = (self.loss_weights['reconstruction'] * loss_1 + 
                            self.loss_weights['alignment'] * loss_2 + 
                            self.loss_weights['crossmodal'] * loss_3)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)


        # Save models
        self.save_model(self.encoder, 'saved_models/bs_encoder_weights_100ep_allaugsimmetry.pth')
        self.save_model(self.projector, 'saved_models/txt_proj_weights_100ep_allaugsimmetry.pth')
        self.save_model(self.decoder, 'saved_models/bs_decoder_weights_100ep_allaugsimmetry.pth')