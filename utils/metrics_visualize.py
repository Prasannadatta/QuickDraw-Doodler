import matplotlib.pyplot as plt
import json
import os

def plot_generator_metrics(metrics, cur_time, cur_epoch, save_dir):
    """
    Plots training and validation metrics as subplots (2x3 grid).
    Each metric is plotted against the number of epochs.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, cur_epoch + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Generator Metrics Over Epochs", fontsize=16)

    # Train and validation losses
    axes[0, 0].plot(epochs, metrics['train']['total_loss'], label='Train')
    axes[0, 0].plot(epochs, metrics['val']['total_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, metrics['train']['kl_div_loss'], label='Train')
    axes[0, 1].plot(epochs, metrics['val']['kl_div_loss'], label='Val')
    axes[0, 1].set_title('KL Divergence Loss')
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, metrics['train']['recon_loss'], label='Train')
    axes[0, 2].plot(epochs, metrics['val']['recon_loss'], label='Val')
    axes[0, 2].set_title('Reconstruction Loss')
    axes[0, 2].legend()

    # Latent space metrics
    axes[1, 0].plot(epochs, metrics['train']['latent_variance'], label='Train')
    axes[1, 0].plot(epochs, metrics['val']['latent_variance'], label='Val')
    axes[1, 0].set_title('Latent Variance')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, metrics['train']['latent_smoothness'], label='Train')
    axes[1, 1].plot(epochs, metrics['val']['latent_smoothness'], label='Val')
    axes[1, 1].set_title('Latent Smoothness')
    axes[1, 1].legend()

    # Diversity metric
    axes[1, 2].plot(epochs, metrics['train']['unique_ratio'], label='Train')
    axes[1, 2].plot(epochs, metrics['val']['unique_ratio'], label='Val')
    axes[1, 2].set_title('Unique Output Ratio')
    axes[1, 2].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fn = f"DoodleGenRNN_{cur_time}.png"
    fp = os.path.join(save_dir, fn)
    plt.savefig(fp, dpi=400)

def log_metrics(metrics, cur_time, cur_epoch, log_dir):
    """
    Logs metrics and hyperparameters into a JSON file.

    Args:
        metrics (dict): Training, validation, and test metrics, including hyperparameters.
        log_dir (str): Directory to save the log file.

    Returns:
        str: The path to the saved log file (to associate with the plot).
    """
    os.makedirs(log_dir, exist_ok=True)
    
    filename = f"DoodleGenRNN_{cur_time}.json"
    filepath = os.path.join(log_dir, filename)

    # Write metrics to a JSON file
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics and hyperparameters logged to: {filepath}")
    return filepath

# function created for cnn
def plot_cnn_metrics(metrics, cur_time, cur_epoch, save_dir):
    """
    Plots training and validation metrics for the CNN as subplots (1x2 grid).
    Each metric is plotted against the number of epochs.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, cur_epoch + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("CNN Metrics Over Epochs", fontsize=16)

    # Train and validation losses
    axes[0].plot(epochs, metrics['train']['total_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs, metrics['val']['total_loss'], label='Validation Loss', color='orange')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Train and validation accuracies
    axes[1].plot(epochs, metrics['train']['accuracy'], label='Train Accuracy', color='green')
    axes[1].plot(epochs, metrics['val']['accuracy'], label='Validation Accuracy', color='red')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    fn = f"ClassifierCNN_{cur_time}.png"
    fp = os.path.join(save_dir, fn)
    plt.savefig(fp, dpi=400)
    print(f"Metrics plot saved at: {fp}")    