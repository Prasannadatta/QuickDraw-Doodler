import matplotlib.pyplot as plt
import json
import os

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

def compute_jsd_wd(real_data, gen_data, num_bins=400):
    # Compute histograms
    hist_range = (min(real_data.min(), gen_data.min()), max(real_data.max(), gen_data.max()))
    hist_real, bin_edges = np.histogram(real_data, bins=num_bins, range=hist_range, density=True)
    hist_gen, _ = np.histogram(gen_data, bins=bin_edges, density=True)
    
    # JSD
    hist_real = np.asarray(hist_real, dtype=np.float64)
    hist_gen = np.asarray(hist_gen, dtype=np.float64)
    m = 0.5 * (hist_real + hist_gen)
    jsd = 0.5 * (np.sum(rel_entr(hist_real, m)) + np.sum(rel_entr(hist_gen, m)))
    normalized_jsd = jsd / np.log(2)
    
    # WD
    wd = wasserstein_distance(real_data, gen_data)
    return normalized_jsd, wd, bin_edges, hist_real, hist_gen

def plot_generator_metrics(metrics, cur_time, cur_epoch, save_dir):
    """
    Plots training and validation metrics as subplots (2x3 grid).
    Each metric is plotted against the number of epochs.
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, cur_epoch + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle("Generator Losses Over Epochs", fontsize=16)

    # Train and validation losses
    axes[0].plot(epochs, metrics['train']['total_loss'], label='Train')
    axes[0].plot(epochs, metrics['val']['total_loss'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].legend()

    axes[1].plot(epochs, metrics['train']['kl_div_loss'], label='Train')
    axes[1].plot(epochs, metrics['val']['kl_div_loss'], label='Val')
    axes[1].set_title('KL Divergence Loss')
    axes[1].legend()

    axes[2].plot(epochs, metrics['train']['recon_loss'], label='Train')
    axes[2].plot(epochs, metrics['val']['recon_loss'], label='Val')
    axes[2].set_title('Reconstruction Loss')
    axes[2].legend()

    plt.tight_layout()

    fn = f"DoodleGenRNN-losses-{cur_time}.png"
    fp = os.path.join(save_dir, fn)
    plt.savefig(fp, dpi=400)

def distribution_comparison(fig, ax, var_idx, real_data, gen_data, variable_name, epoch, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    jsd, wd, bin_edges, hist_real, hist_gen = compute_jsd_wd(real_data, gen_data)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # get center of histogram bins
    
    # plot real data as a histogram
    ax[var_idx].hist(
        bin_centers, 
        bins=bin_edges, 
        weights=hist_real, 
        alpha=0.6, 
        color='red', 
        density=False, 
        label=f'{variable_name} Real Data',
    )

    # plot generated data as a line
    ax[var_idx].plot(
        bin_centers, 
        hist_gen, 
        color='black', 
        linewidth=2, 
        label=f'{variable_name} Generated Data'
    )

    # Set labels and title
    ax[var_idx].set_xlabel(variable_name, fontsize=12)
    ax[var_idx].set_ylabel('Count', fontsize=12)
    ax[var_idx].set_title(f'Distribution Comparison: {variable_name} - Epoch {epoch}\nJSD: {jsd:.4f}, WD: {wd:.4f}', fontsize=14)
    
    ax[var_idx].legend(fontsize=12)
    fig.tight_layout() 
    
    return fig, ax, jsd, wd


def log_metrics(metrics, filename, log_dir):
    """
    Logs metrics and hyperparameters into a JSON file.

    Args:
        metrics (dict): Training, validation, and test metrics, including hyperparameters.
        log_dir (str): Directory to save the log file.

    Returns:
        str: The path to the saved log file (to associate with the plot).
    """
    os.makedirs(log_dir, exist_ok=True)
    
    filepath = os.path.join(log_dir, filename)

    # Write metrics to a JSON file
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics logged to: {filepath}")
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
    axes[0].plot(epochs, metrics['train']['loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs, metrics['val']['loss'], label='Validation Loss', color='orange')
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
    fn = f"ClassifierCNN_{cur_epoch}_{cur_time}.png"
    fp = os.path.join(save_dir, fn)
    plt.savefig(fp, dpi=400)
    print(f"Metrics plot saved at: {fp}")    