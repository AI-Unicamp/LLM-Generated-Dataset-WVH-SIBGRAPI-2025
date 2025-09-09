import pandas as pd
from dataset import BlendshapeDataset_all, BlendshapeDataset_one
from trainer import Trainer
from model import BlendshapeEncoder, BlendshapeDecoder, ClipEncoderModule, TextProjector

# Load your dataframe containing the blendshapes data
dataframe = pd.read_csv('./gen_data/dataset_with_BS.csv')
dataframe_test = dataframe[:5]

# Create the dataset
dataset_one = BlendshapeDataset_one(dataframe)
dataset_all = BlendshapeDataset_all(dataframe)

# Initialize the models
encoder = BlendshapeEncoder()
decoder = BlendshapeDecoder()
projector = TextProjector()
clip_encoder = ClipEncoderModule()

# Create the trainer and start training
trainer = Trainer(encoder=encoder, 
                  decoder=decoder,
                  projector=projector,
                  clip_encoder=clip_encoder, 
                  dataset=dataset_all, 
                  batch_size=256, 
                  learning_rate=1e-5, 
                  epochs=200, 
                  device='cuda:0')
trainer.train()
