import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset  # add subset import for dataset restriction
from torchvision import transforms
from utils import load_config
from model_fnn import VariationalAutoEncoder
from model_cnn import ConvolutionalVAE
from trainer import Trainer

# ================ BASELINE PREPROCESSING AND DATA LOADING ========================
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
# =================================================================================


# ======================= DATA AUGMENTATION FOR TRAINING ==========================
#train_transform = transforms.Compose([
#    transforms.RandomRotation(
#        degrees=15
#    ),  # Rotate randomly between -15 and +15 degrees
#        # Have to avoid flips/mirrors because they could change the digit's meaning 
#        # (e.g., 6 becomes 9, 8 stays the same, etc.)
#    transforms.ToTensor(),
#])
#
## Keep test transform clean (no random augmentation during evaluation!)
#test_transform = transforms.Compose([transforms.ToTensor()])
#
## Load datasets with respective transforms
#train_dataset = datasets.MNIST(
#    root='data', train=True, transform=train_transform, download=True
#)
#
#test_dataset = datasets.MNIST(
#    root='data', train=False, transform=test_transform, download=True
#)
# =================================================================================


# ================ RESTRICT DATASET SIZE HERE =====================================
#subset_size = 5000  # Change this to your desired size
#train_dataset = Subset(train_dataset, list(range(subset_size)))
#print (len(train_dataset)) # ensure the dataset is restricted to the desired size
# =================================================================================


test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    target_transform=None,
    download=True
)

if __name__ == "__main__":
    
    config = load_config("config.yaml")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # <--- Uses Mac's GPU
    else:
        device = torch.device("cpu")

    #model = VariationalAutoEncoder(
    #    input_dim=config['model']['input_dim'],
    #    latent_dim=config['model']['latent_dim'],
    #    activation=config['model']['activation'],
    #)

    model = ConvolutionalVAE(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        activation=config['model']['activation'],
    )

    # Check which model we're using
    print(f"Successfully initialized model: {type(model).__name__}")
    
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
