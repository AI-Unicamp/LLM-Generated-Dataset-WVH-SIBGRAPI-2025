
import pandas as pd
import torch
import ast
import numpy as np
from model_clip import BlendshapeEncoder, TextProjector, ClipEncoderModule, BlendshapeDecoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import KDTree

class EmbeddingGenerator:
    def __init__(self, df, bs_encoder_path=None,  
                 text_projector_path=None, 
                 clip_encoder=None,
                 bs_decoder_path=None,
                 device=None):
        """
        Initialize the EmbeddingGenerator with trained models.
        
        Args:
            df: DataFrame containing blendshape data
            bs_encoder_path: Path to the saved blendshape encoder weights
            text_projector_path: Path to the saved text projector weights
            clip_encoder: CLIP encoder class
            bs_decoder_path: Path to the saved blendshape decoder weights
            device: Device to run the model on
        """
        self.device = device
        self.bs_encoder = self.load_model(bs_encoder_path, BlendshapeEncoder)
        self.text_projector = self.load_model(text_projector_path, TextProjector)
        self.bs_decoder = self.load_model(bs_decoder_path, BlendshapeDecoder)
        self.clip_encoder = clip_encoder().to(self.device)
        self.df = df.copy()
        
        # Fill NaN values to prevent errors
        text_columns = ['Sentences', 'Emotions', 'Description']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)
        
    def load_model(self, model_weights, model_class):
        """
        Load a model from saved weights.
        
        Args:
            model_weights: Path to the saved model weights
            model_class: Class of the model to instantiate
            
        Returns:
            Loaded model set to evaluation mode
        """
        model = model_class() 
        model.load_state_dict(torch.load(model_weights, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set to evaluation mode
        print(f"Loaded model weights from {model_weights}")
        return model

    def process_text_embedding(self, text):
        """
        Process text through CLIP encoder and text projector safely.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of the embedding
        """
        # ClipEncoderModule expects a list of strings
        if not isinstance(text, str) or not text:
            text = ""
            
        # Pass a list containing the single string to match ClipEncoderModule's expectations
        clip_output = self.clip_encoder([text])
        embedding = self.text_projector(clip_output)
        return embedding.detach().cpu().numpy().flatten()

    def get_embeddings(self):
        """
        Extract all types of embeddings from the dataframe.
        
        Returns:
            Dictionary containing all embeddings and their labels
        """
        t_embedding_list = []
        e_embedding_list = []
        d_embedding_list = []

        with torch.no_grad():
            total_rows = len(self.df)
            for idx, row in self.df.iterrows():
                if (idx + 1) % 100 == 0 or idx == 0:
                    print(f'Processing {idx+1}/{total_rows} rows ({(idx+1)/total_rows*100:.1f}%)')
                
                try:
                    # Process text data
                    t_embedding_np = self.process_text_embedding(row['Sentence'])
                    t_embedding_list.append(t_embedding_np)
                    
                    e_embedding_np = self.process_text_embedding(row['Emotions'])
                    e_embedding_list.append(e_embedding_np)
                    
                    d_embedding_np = self.process_text_embedding(row['Description'])
                    d_embedding_list.append(d_embedding_np)
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    # Skip this row and continue with the next one
                    continue

        # Verify we have data to work with
        if not t_embedding_list:
            raise ValueError("No texts embeddings were generated. Check your input data and model.")
        if not e_embedding_list:
            raise ValueError("No emotion embeddings were generated. Check your input data and model.")
        if not d_embedding_list:
            raise ValueError("No description embeddings were generated. Check your input data and model.")
            
        return {
            'Sentence': np.array(t_embedding_list),
            'Emotions': np.array(e_embedding_list),
            'Description': np.array(d_embedding_list)
        }

    def apply_tsne_to_all(self, embeddings_dict, perplexity=50, learning_rate=200, random_state=42):
        """
        Apply t-SNE to all embedding types.
        
        Args:
            embeddings_dict: Dictionary of embedding arrays
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing t-SNE results for each embedding type
        """
        tsne_results = {}
        
        for name, embeddings in embeddings_dict.items():
            print(f"Applying t-SNE to {name} embeddings (shape: {embeddings.shape})...")
            
            # Check if we have enough samples for the current perplexity
            current_perplexity = min(perplexity, embeddings.shape[0] - 1)
            if current_perplexity != perplexity:
                print(f"Warning: Reducing perplexity to {current_perplexity} for {name} due to sample size")
                
            tsne = TSNE(n_components=2, 
                        perplexity=current_perplexity, 
                        learning_rate=learning_rate, 
                        random_state=random_state,
                        n_jobs=-1)  # Use all available cores
            tsne_results[name] = tsne.fit_transform(embeddings)
            
        return tsne_results
    
    def plot_tsne_comparison(self, tsne_results, figsize=(15, 12), save_path='tsne_comparison.png'):
        """
        Plot t-SNE results with different colors for each embedding type.
        
        Args:
            tsne_results: Dictionary of t-SNE results
            figsize: Figure size
            save_path: Path to save the figure
        """
        # Define custom colors for each embedding type
        colors = {
            'Sentence': '#6BAED6',  # Light blue
            'Emotions': '#C59EC9',  # Purple
            'Description': '#F8A19F'  # Light pink
        }
        markers = {'Sentence': 'o', 'Emotions': 's', 'Description': 'D'}
        
        # Define display names for the legend
        display_names = {
            'Sentence': 'Transcripts',
            'Emotions': 'Emotions',
            'Description': 'Descriptions'
        }
        
        plt.figure(figsize=figsize)
        
        # Plot each embedding type
        for name, tsne_data in tsne_results.items():
            plt.scatter(tsne_data[:, 0], 
                        tsne_data[:, 1],
                        c=colors[name],
                        marker=markers[name],
                        s=50,
                        alpha=0.7,
                        label=display_names[name],  # Use display name for legend
                        edgecolors='none')
        
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.title('Comparison of Different Embedding Types using t-SNE', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()

    def create_arrow_paths(self, tsne_results, embeddings_dict, n_lines=3, n_points=4, 
                    central_region_pct=0.6, save_path='arrow_paths.png',
                    line_names=None, mixed_embeddings=True):
        """
        Create arrow paths in the central region of t-SNE space and return decoded blendshapes.
        
        Args:
            tsne_results: Dictionary of t-SNE results
            embeddings_dict: Dictionary of original embeddings
            n_lines: Number of lines to generate
            n_points: Number of points per line (used for internal calculations only)
            central_region_pct: Percentage of the plot to consider as central region (0-1)
            save_path: Path to save the visualization
            line_names: Custom names for each line
            mixed_embeddings: Whether to mix embedding types for each line
            
        Returns:
            Dictionary with decoded blendshape vectors for each line
        """
        # Create a figure for the combined plot
        plt.figure(figsize=(15, 12))
        
        # Updated colors for the different embedding types
        embed_colors = {
            'Sentence': '#6BAED6',  # Light blue
            'Emotions': '#C59EC9',  # Purple
            'Description': '#F8A19F'  # Light pink
        }
        embed_markers = {'Sentence': 'o', 'Emotions': 's', 'Description': 'D'}
        
        # Define display names for the legend
        display_names = {
            'Sentence': 'Transcripts',
            'Emotions': 'Emotions',
            'Description': 'Descriptions'
        }
        
        # If no custom line names provided, use defaults
        if line_names is None:
            line_names = [f"Path {i+1}" for i in range(n_lines)]
        elif len(line_names) < n_lines:
            # Extend with default names if not enough provided
            line_names.extend([f"Path {i+1}" for i in range(len(line_names), n_lines)])
        
        # Merge all TSNE results to calculate the overall boundaries
        all_points = np.vstack([data for data in tsne_results.values()])
        
        # Calculate min and max values for each axis
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        # Calculate the central region boundaries (to avoid outliers)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        central_x_min = x_min + (1 - central_region_pct) / 2 * x_range
        central_x_max = x_max - (1 - central_region_pct) / 2 * x_range
        central_y_min = y_min + (1 - central_region_pct) / 2 * y_range
        central_y_max = y_max - (1 - central_region_pct) / 2 * y_range
        
        # Plot all embedding points with reduced alpha
        for embed_type, tsne_data in tsne_results.items():
            plt.scatter(tsne_data[:, 0], 
                    tsne_data[:, 1],
                    c=embed_colors[embed_type],
                    marker=embed_markers[embed_type],
                    s=30,
                    alpha=0.2,  # More translucent
                    label=display_names[embed_type])
        
        # Initialize dictionary to store decoded blendshapes
        all_decoded_blendshapes = {}
        
        # Define start points for arrows - focused on central region
        # These angles create arrows pointing in different directions
        angles = np.linspace(0, 2*np.pi, n_lines, endpoint=False)
        central_x = (central_x_min + central_x_max) / 2
        central_y = (central_y_min + central_y_max) / 2
        
        # Calculate central radius - distance from center to edge of central region
        radius_x = (central_x_max - central_x_min) / 2 * 0.8  # 80% of half-width
        radius_y = (central_y_max - central_y_min) / 2 * 0.8  # 80% of half-height
        radius = min(radius_x, radius_y)
        
        # Generate line paths
        for line_idx, angle in enumerate(angles):
            # Calculate arrow direction (unit vector)
            dx = np.cos(angle)
            dy = np.sin(angle)
            direction = np.array([dx, dy])
            
            # Pick an offset from center for the starting point (30-50% toward edge)
            offset_factor = np.random.uniform(0.3, 0.5)
            start_point = np.array([
                central_x + offset_factor * radius * dx,
                central_y + offset_factor * radius * dy
            ])
            
            # Calculate arrow length (40-60% of remaining distance to edge)
            length_factor = np.random.uniform(2.0, 2.2)
            arrow_length = (1 - offset_factor) * radius * length_factor
            
            # Calculate end point
            end_point = start_point + direction * arrow_length
            
            # Generate n_points along the arrow (for calculations only, not displayed)
            line_points = []
            for i in range(n_points):
                # Calculate position along line
                t = i * arrow_length / (n_points - 1)
                
                # Get point coordinates
                point = start_point + t * direction
                line_points.append(point)
            
            line_points = np.array(line_points)
            
            # Draw arrow - make the arrows more prominent since they're the focus now
            arrow = FancyArrowPatch(
                start_point, end_point,
                arrowstyle='-|>',
                mutation_scale=25,  # Increased size of the arrow head for better visibility
                linewidth=3,  # Thicker line
                color='black',
                alpha=0.9  # Higher opacity
            )
            plt.gca().add_patch(arrow)
            
            # Dictionary to store decoded blendshapes for this line
            line_decoded = {}
            
            # For each point, use a custom sequence of embedding types
            for point_idx, point in enumerate(line_points):
                # Instead of cycling through embedding types automatically,
                # use our custom sequence: Description, Emotions, Sentence, Emotions
                custom_sequence = ['Emotions', 'Emotions', 'Emotions', 'Emotions']
                embed_type = custom_sequence[point_idx % len(custom_sequence)]
                
                # Find the closest point in this embedding space
                tsne_data = tsne_results[embed_type]
                tree = KDTree(tsne_data)
                dist, idx = tree.query(point)
                
                # Get the original embedding
                original_embedding = embeddings_dict[embed_type][idx]
                
                # Convert to torch tensor
                embedding_tensor = torch.tensor(original_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Decode to blendshapes
                with torch.no_grad():
                    decoded_blendshape = self.bs_decoder(embedding_tensor)
                    
                # Convert to numpy
                decoded_np = decoded_blendshape.cpu().numpy().flatten()
                
                # Store with source information
                line_decoded[f"Point_{point_idx+1}"] = {
                    'embedding_type': embed_type,
                    'original_index': idx,
                    'blendshape': decoded_np,
                    'tsne_coords': point
                }
            
            # Store all results for this line
            all_decoded_blendshapes[line_names[line_idx]] = line_decoded
        
        plt.xlabel('t-SNE Component 1', fontsize=28)
        plt.ylabel('t-SNE Component 2', fontsize=28)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set layout with extra padding to avoid legend cutoff
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Leave space for the legend
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Arrow paths visualization saved to {save_path}")
        plt.show()
        
        return all_decoded_blendshapes



# Initialize the enhanced EmbeddingGenerator with decoder path
device = "cuda:7"  # Use the same device you had in your original code
    
# Model paths
bs_encoder_path = 'saved_models/bs_encoder_weights_200ep_allaug_EDA.pth'  
text_proj_path = 'saved_models/txt_proj_weights_200ep_allaug_EDA.pth'
bs_decoder_path = 'saved_models/bs_decoder_weights_200ep_allaug_EDA.pth'

df = pd.read_csv('dataset/bs_llama3370b_fined_final_right.csv')
        
# Print dataframe info for debugging
print(f"DataFrame loaded: {len(df)} rows, columns: {df.columns.tolist()}")

#10681

# Sample data for faster processing
#sample_size = min(5000, len(df))  # Adjust as needed
sample_df = df.sample(n=10681, random_state=42)

generator = EmbeddingGenerator(
    df=sample_df,
    bs_encoder_path=bs_encoder_path,
    text_projector_path=text_proj_path,
    clip_encoder=ClipEncoderModule,
    bs_decoder_path=bs_decoder_path,
    device=torch.device(device if torch.cuda.is_available() else "cpu")
)

# Get embeddings
embeddings_dict = generator.get_embeddings()

# Apply t-SNE
tsne_results = generator.apply_tsne_to_all(embeddings_dict)

# Find 3 lines with 4 points each and get the decoded blendshapes
decoded_blendshapes = generator.create_arrow_paths(
    tsne_results,
    embeddings_dict,
    n_lines=3,
    n_points=4,
    central_region_pct=0.6,  # Focus on central 60% of the plot
    save_path='tsne_plots/central_arrow_paths_0.2data_allaug_EDA_fined_alle_test2.png',
    mixed_embeddings=True  # Use mixed embedding types along each path
)

# Print the blendshape vectors for each point
for line_name, line_data in decoded_blendshapes.items():
    print(f"\n=== {line_name} ===")
    for point_name, point_data in line_data.items():
        embedding_type = point_data['embedding_type']
        print(f"  {point_name} ({embedding_type}):")
        print(f"  Blendshape: {point_data['blendshape']}")
        print()