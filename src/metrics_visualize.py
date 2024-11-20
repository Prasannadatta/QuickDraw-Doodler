import matplotlib.pyplot as plt
import json
import os

def plot_generator_metrics(metrics, cur_time, save_dir):
    """
    Plots training and validation metrics as subplots (2x3 grid).
    Each metric is plotted against the number of epochs.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = metrics['hyperparams']["num_epochs"]
    epochs = range(1, num_epochs + 1)
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

def log_metrics(metrics, cur_time, log_dir):
    """
    Logs metrics and hyperparameters into a JSON file.

    Args:
        metrics (dict): Training, validation, and test metrics, including hyperparameters.
        log_dir (str): Directory to save the log file.

    Returns:
        str: The path to the saved log file (to associate with the plot).
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Construct filename using final metrics and datetime for clarity
    filename = (
        f"DoodleGenRNN_{cur_time}_metrics-log_loss-{metrics['train']['total_loss'][-1]}_"
        f"latentvar-{metrics['train']['latent_variance'][-1]}.json"
    )
    filepath = os.path.join(log_dir, filename)

    # Write metrics to a JSON file
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics and hyperparameters logged to: {filepath}")
    return filepath