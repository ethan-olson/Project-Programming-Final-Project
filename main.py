import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset  # add subset import for dataset restriction
from torchvision import transforms
from utils import load_config
from model_fnn import VariationalAutoEncoder
from trainer import Trainer

# Data preprocessing with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Baseline dataset loader
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    target_transform=None,
    download=True
)

# --- RESTRICT DATASET SIZE HERE ---
subset_size = 5000  # Change this to your desired size
train_dataset = Subset(train_dataset, list(range(subset_size)))
print (len(train_dataset)) # ensure the dataset is restricted to the desired size
# ----------------------------------


test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    target_transform=None,
    download=True
)

if __name__ == "__main__":
    
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariationalAutoEncoder(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        activation=config['model']['activation'],
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_path=None,
        experiment_name="VAE-MNIST",
        logger_step = 100,
        output_dir="outputs"
    )

    trainer.train(train_dataset=train_dataset, val_dataset=test_dataset)
