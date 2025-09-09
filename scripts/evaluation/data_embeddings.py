'''import pandas as pd
import torch
import ast
import numpy as np
from model_claude import BlendshapeEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EmbeddingGenerator:
    def __init__(self, df, encoder, device="cuda:1"):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.df = df.copy()
        self.df["bs_embeddings"] = None

    def get_embeddings(self):
        embeddings_list = []

        with torch.no_grad():
            for idx, row in self.df.iterrows():
                if isinstance(row["blendshapes"], str):
                    blendshapes = torch.tensor(ast.literal_eval(row["blendshapes"]), dtype=torch.float32)
                else:
                    blendshapes = torch.tensor(row["blendshapes"], dtype=torch.float32)

                blendshapes = blendshapes[1:]

                blendshapes = blendshapes.to(self.device)

                embedding = self.encoder(blendshapes.unsqueeze(0))
                embedding_np = embedding.detach().cpu().numpy().flatten()
                embeddings_list.append(embedding_np)

                print(f'Number of samples processed: {len(embeddings_list)}')

        return np.array(embeddings_list)  # Return as a 2D array

    def apply_tsne(self, embeddings, perplexity=30, learning_rate=200, random_state=42):
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, random_state=random_state)
        return tsne.fit_transform(embeddings)
    
    def plot_tsne(self, tsne_results, figsize=(12, 8), save_path='dataset/tsne_plot_10000'):
        plt.figure(figsize=figsize)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Blendshape Embeddings t-SNE Visualization')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()

# Example Usage
encoder = BlendshapeEncoder()
df = pd.read_csv('dataset/bs_llm_llama32_3b_all.csv')

embed_generator = EmbeddingGenerator(df.sample(n=10000, random_state=42), encoder=encoder)
embeddings = embed_generator.get_embeddings()

# Apply t-SNE
tsne = embed_generator.apply_tsne(embeddings)

# Plot and save the t-SNE results
embed_generator.plot_tsne(tsne)'''


import pandas as pd
import torch
import ast
import numpy as np
from model_clip import BlendshapeEncoder, TextProjector, ClipEncoderModule
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EmbeddingGenerator:
    def __init__(self, df, bs_encoder_path=None,  
                 text_projector_path=None, 
                 clip_encoder=None,
                 device=None):
        """
        Initialize the EmbeddingGenerator with a trained model.
        
        Args:
            df: DataFrame containing blendshape data
            model_path: Path to the saved model weights (.pth or .pt file)
            device: Device to run the model on (default: "cuda:1")
        """
        self.device = device
        self.bs_encoder = self.LoadModel(bs_encoder_path, BlendshapeEncoder)
        self.text_projector = self.LoadModel(text_projector_path, TextProjector)
        self.clip_encoder = clip_encoder()
        self.df = df.copy()
        self.df["t_embeddings"] = None
        self.df["e_embeddings"] = None
        self.df["d_embeddings"] = None
        self.df["au_embeddings"] = None
        self.df["bs_embeddings"] = None

    '''def _load_model(self, model_path):
        """
        Load the trained model weights and prepare the encoder.
        
        Args:
            model_path: Path to the saved model weights
            
        Returns:
            Loaded and prepared encoder model
        """
        try:
            bs_encoder = BlendshapeEncoder()
            text_proj = TextProjector()
            #clip_encoder = ClipEncoderModule()

            encoder.load_state_dict(torch.load(model_path, map_location=self.device))
            encoder.eval()  # Set to evaluation mode
            encoder = encoder.to(self.device)
            print(f"Successfully loaded model from {model_path}")
            return encoder
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")'''
        
    def LoadModel(self, model_weights, model_class):
        self.model = model_class() 
        self.model.load_state_dict(torch.load(model_weights))
        self.model.to(self.device)
        #self.logger.info(f"Loaded model weights from {model_weights}")

    def get_embeddings(self):
        bs_embbeding_list = []
        t_embbeding_list = []
        e_embbeding_list = []
        d_embbeding_list = []
        au_embbeding_list = []

        with torch.no_grad():
            for idx, row in self.df.iterrows():
                if isinstance(row["blendshapes"], str):
                    blendshapes = torch.tensor(ast.literal_eval(row["blendshapes"]), dtype=torch.float32)
                else:
                    blendshapes = torch.tensor(row["blendshapes"], dtype=torch.float32)

                blendshapes = blendshapes[1:]
                blendshapes = blendshapes.to(self.device)

                t,e,d,au = row['Sentence'], row['Emotions'], row['Description'], row['Action Units']

                t_embedding = self.text_projector(self.clip_encoder(t))
                t_embedding_np = t_embedding.detach().cpu().numpy().flatten()
                t_embbeding_list.append(t_embedding_np)

                e_embedding = self.text_projector(self.clip_encoder(e))
                e_embedding_np = e_embedding.detach().cpu().numpy().flatten()
                e_embbeding_list.append(e_embedding_np)

                d_embedding = self.text_projector(self.clip_encoder(d))
                d_embedding_np = d_embedding.detach().cpu().numpy().flatten()
                d_embbeding_list.append(d_embedding_np)

                au_embedding = self.text_projector(self.clip_encoder(au))
                au_embedding_np = au_embedding.detach().cpu().numpy().flatten()
                au_embbeding_list.append(au_embedding_np)
                

                bs_embedding = self.encoder(blendshapes.unsqueeze(0))
                bs_embedding_np = bs_embedding.detach().cpu().numpy().flatten()
                bs_embbeding_list.append(bs_embedding_np)

                if (len(bs_embbeding_list) % 1000) == 0:
                    print(f'Number of samples processed: {len(bs_embbeding_list)}')

        return np.array(t_embbeding_list), np.array(e_embbeding_list), np.array(d_embbeding_list), np.array(au_embbeding_list), np.array(bs_embbeding_list), 

    def apply_tsne(self, embeddings, perplexity=30, learning_rate=200, random_state=42):
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, random_state=random_state)
        return tsne.fit_transform(embeddings)
    
    def plot_tsne(self, tsne_results, figsize=(12, 8), save_path='dataset/tsne_plot_53k_aug_50ep'):
        plt.figure(figsize=figsize)
        plt.scatter(tsne_results[:, 0], 
                    tsne_results[:, 1],
                    linewidths=0, 
                    alpha=0.6)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Blendshape Embeddings t-SNE Visualization')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()

device = "cuda:7"

# Example Usage
bs_encoder_path = 'saved_models/bs_encoder_weights_aug1.pth'  
text_proj_path = 'saved_models/txt_proj_weights_aug1.pth' 
df = pd.read_csv('dataset/bs_llm_llama32_3b_all.csv')

# Initialize with trained model
embed_generator = EmbeddingGenerator(
    df=df.sample(n=10000, random_state=42),
    bs_encoder_path=bs_encoder_path, 
    text_projector_path=text_proj_path, 
    clip_encoder=ClipEncoderModule, 
    device=device 
)

# Generate embeddings using trained model
embeddings = embed_generator.get_embeddings()

# Apply t-SNE
tsne = embed_generator.apply_tsne(embeddings)

# Plot and save the t-SNE results
embed_generator.plot_tsne(tsne)
