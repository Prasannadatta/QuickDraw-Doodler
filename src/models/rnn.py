import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import os

from src.process_data import init_sequential_dataloaders
from src.metrics_visualize import plot_generator_metrics, log_metrics
    
class DoodleGenRNN(nn.Module):
    def __init__(self, in_size, hidden_size, latent_size, num_layers, num_labels, dropout=0.2, use_fc_activations=False):
        super(DoodleGenRNN, self).__init__()

        self.use_fc_activations = use_fc_activations

        # encode sequential stroke data with bidirectional LSTM
        # final hidden state of lstm defines the latent space
        self.lstm_encoder = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True, # data is loaded with batch dim first (b, seq_len, points)
            bidirectional=True
        )

        # latent space which will be pulled from for generation
        # latent space -> compressed representation of input data to lower dimensional space
        # like convolutions grab most important features, this latent space will learn the same
        # essentially hidden lstm to latent space, but with reparameterizing to sample from gaussian dist of mean and var of hidden
        self.hidden_to_mu_fc = nn.Linear(hidden_size * 2, latent_size) # * 2 for bidirectionality
        self.hidden_to_logvar_fc = nn.Linear(hidden_size * 2, latent_size)

        # embedding for the labels
        # maps discrete labels (idxs for sketch labels e.g. 1 for cat, 2 for tree),
        # to learned cts dense vectors e.g. 1 -> [0.2, 0.9, 0.34,...] to size of latent space (latent_size)
        # allows model to learn relationship between label idxs and the input data
        self.label_embedding = nn.Embedding(num_labels, latent_size)

        # decoder LSTM
        self.latent_to_hidden_fc = nn.Linear(latent_size * 2, hidden_size)
        self.lstm_decoder = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        self.hidden_to_output_fc = nn.Linear(hidden_size, in_size)

        self.dropout_fc = nn.Dropout(dropout) # dropout layer

    def encode(self, x):
        """
        Encode input sequence of strokes into latent space 
        Final hidden state of lstm is the encoded sequence (compressed/summarized input)

        """
        # grab final hidden state of lstm encoder
        _, (hn, _) = self.lstm_encoder(x)

        # since our lstm is bidirectional we grab the forward and backward states
        hidden = torch.cat((hn[-2], hn[-1]), dim=1)

        # mu and logvar aren't the actual mean and log of variance of hidden state,
        # rather they're learned params we will use as the "mean" and log of "variance" to sample z from
        mu = self.hidden_to_mu_fc(hidden) # mean of latent vec (z)
        logvar = self.hidden_to_logvar_fc(hidden) # log of variance of latent vec, use log for numerical stability

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize: Sample z ~ N(mu, sigma^2)
        essentially sampling the latent space vector (z),
        from normal distribution defined by stats of encoder's hidden state
        
        allows for backpropagation of network,
        since directly sampling as described above is non-differentiable,
        due to the randomness involved in sampling,
        but reformulating the sampling is since epsilon contains the randomness,
        and is not part of the model's parameters.
        """
        sigma = torch.exp(0.5 * logvar) # standard deviation
        epsilon = torch.randn_like(sigma) # epsilon ~ N(0,1) -> random sample from normal dis
        z = mu + epsilon * sigma # adds some random var to latent space to increase generation diversity

        return z

    def decode(self, z, seq_len, labels, inputs=None):
        """
        Decode the encoders's output while conditioning the latent vec on the embedded label
        """
        # learn a relationship between label and sample
        # only giving label of one class per sample, labels is plural due to batch processing
        # embedding learns relations indirectly since it is involved in the network,
        # but does not directly receive input sequences to associate with labels
        label_embedding = self.label_embedding(labels)
        
        # give latent vector z the condition
        z_conditioned = torch.cat((z, label_embedding), dim=1)

        batch_size = z.size(0)

        # prepare hidden state for decoder mapping from z to generate initial hidden state
        # unsqueeze to add dim of num_layers for decoder lstm since it expects (num_layers (1 initially), B, hidden_size)
        hidden = self.latent_to_hidden_fc(z_conditioned).unsqueeze(0)
        hidden = F.relu(hidden) if self.use_fc_activations else hidden

        # init hidden state from z that is mapped from the above fc layer is repeated across lstm layers
        h0 = hidden.repeat(self.lstm_decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0) # initial cell state set to 0
        hidden = (h0, c0)

        if inputs is None: # if being asked to generate without conditional label
            # unconditional generation inputs are 0s
            inputs = torch.zeros(batch_size, seq_len, self.hidden_to_output_fc.out_features).to(z.device)

        # decoding start
        # pass entire sequence through decoder
        decode_out, _ = self.lstm_decoder(inputs, hidden)

        # decoder output gets mapped to original input space
        outputs = self.hidden_to_output_fc(decode_out)

        return outputs

    def forward(self, x, seq_len, labels, inputs=None):
        mu, logvar = self.encode(x) # encode input sequence
        z = self.reparameterize(mu, logvar) # sample latent vector from encoded sequence
        out = self.decode(z, seq_len, labels, inputs) # decode latent vector with label condition
        
        return out, mu, logvar

def kl_div(Xbatch, mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Xbatch.size(0)

def find_latent_smoothness(all_mus, num_pairs=20, steps=5):
    if len(all_mus) < 2: # can only calculate distances between at least 2 points
        return 0  

    all_mus = torch.cat(all_mus, dim=0)
    n_mus = all_mus.shape[0]

    # randomly sample num_pairs idx pairs
    idx1 = torch.randint(0, n_mus, (num_pairs,))
    idx2 = torch.randint(0, n_mus, (num_pairs,))
    z1 = all_mus[idx1]
    z2 = all_mus[idx2]

    # steps of interpolations
    alphas = torch.linspace(0, 1, steps).to(all_mus.device).view(-1, 1, 1)  # (steps, 1, 1)

    # interpolate between z1 and z2 for all num_pairs
    interpolations = (1 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)  # (steps, num_pairs, latent_dim)

    # euclidean distance between interpolations
    differences = interpolations[1:] - interpolations[:-1]  # dists between consecutive interpolations
    distances = torch.norm(differences, dim=-1)  # norm across latent dimensions
    smoothness = distances.mean().item()  # mean dist as smoothness score

    return smoothness

def train(
        epoch,
        num_epochs,
        train_loader,
        rnn,
        optim,
        reconstruction_criterion,
        alpha, # scale how important kl_div loss is
        device,
        metrics
):
    rnn.train()

    # init metrics
    running_recon_train_loss, running_div_train_loss, running_total_train_loss = 0., 0., 0.
    latent_variances, unique_outputs, all_train_mus = [], [], []
    
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (Xbatch, seq_lens, ybatch) in train_bar:
        # move data to gpu
        Xbatch, seq_lens, ybatch = Xbatch.to(device), seq_lens.to(device), ybatch.to(device)

        # forward
        outputs, mu, logvar = rnn.forward(Xbatch, Xbatch.size(1), ybatch)
        all_train_mus.append(mu.detach().cpu()) # store for latent smoothness metric later

        # loss computation
        rec_loss = reconstruction_criterion(outputs, Xbatch) # basically mse loss between gen and real seq
        kl_div_loss = kl_div(Xbatch, mu, logvar) # how well does generated dis match real data dist
        loss = rec_loss + alpha * kl_div_loss # total loss
        
        # backward
        optim.zero_grad() # zero gradients from prev grad calculation
        loss.backward() # back propagation to find gradients
        optim.step() # step down gradient

        # metrics
        # losses
        running_total_train_loss += loss.item()
        running_div_train_loss += kl_div_loss.item()
        running_recon_train_loss += rec_loss.item()

        latent_variances.append(torch.var(mu, dim=0).mean().item()) # latent space variance

        unique_count = len(torch.unique(outputs, dim=0)) # diversity (unique outputs)
        unique_outputs.append(unique_count / outputs.size(0)) # diversity ratio to total outputs
        
        train_bar.set_postfix(
            total_loss=running_total_train_loss / (batch_idx + 1),
            kl_loss=running_div_train_loss / (batch_idx + 1),
            recon_loss=running_recon_train_loss / (batch_idx + 1)
        )

    # log metrics
    n = len(train_loader)
    metrics['train']['total_loss'].append(running_total_train_loss / n)
    metrics['train']['kl_div_loss'].append(running_div_train_loss / n)
    metrics['train']['recon_loss'].append(running_recon_train_loss / n)
    metrics['train']['latent_variance'].append(sum(latent_variances) / len(latent_variances))
    metrics['train']['latent_smoothness'].append(find_latent_smoothness(all_train_mus))
    metrics['train']['unique_ratio'].append(sum(unique_outputs) / len(unique_outputs))

def validate(
        epoch,
        num_epochs,
        val_loader,
        rnn,
        reconstruction_criterion,
        alpha, # scale how important kl_div loss is
        device,
        metrics
):
    rnn.eval()

    # init metrics
    running_recon_val_loss, running_div_val_loss, running_total_val_loss = 0., 0., 0.
    latent_variances, unique_outputs, all_val_mus = [], [], []
    
    val_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validating Epoch {epoch+1}/{num_epochs}")
    with torch.no_grad():
        for batch_idx, (Xbatch, seq_lens, ybatch) in val_bar:
            # move data to gpu
            Xbatch, seq_lens, ybatch = Xbatch.to(device), seq_lens.to(device), ybatch.to(device)

            # forward
            outputs, mu, logvar = rnn.forward(Xbatch, Xbatch.size(1), ybatch)
            all_val_mus.append(mu.detach().cpu()) # store for latent smoothness metric later

            # loss computation
            rec_loss = reconstruction_criterion(outputs, Xbatch) # basically mse loss between gen and real seq
            kl_div_loss = kl_div(Xbatch, mu, logvar) # how well does generated dis match real data dist
            loss = rec_loss + alpha * kl_div_loss # total loss

            # metrics
            # losses
            running_total_val_loss += loss.item()
            running_div_val_loss += kl_div_loss.item()
            running_recon_val_loss += rec_loss.item()

            latent_variances.append(torch.var(mu, dim=0).mean().item()) # latent space variance

            unique_count = len(torch.unique(outputs, dim=0)) # diversity (unique outputs)
            unique_outputs.append(unique_count / outputs.size(0)) # diversity ratio to total outputs
            
            val_bar.set_postfix(
                total_loss=running_total_val_loss / (batch_idx + 1),
                kl_loss=running_div_val_loss / (batch_idx + 1),
                recon_loss=running_recon_val_loss / (batch_idx + 1)
            )

    # log metrics
    n = len(val_loader)
    metrics['val']['total_loss'].append(running_total_val_loss / n)
    metrics['val']['kl_div_loss'].append(running_div_val_loss / n)
    metrics['val']['recon_loss'].append(running_recon_val_loss / n)
    metrics['val']['latent_variance'].append(sum(latent_variances) / len(latent_variances))
    metrics['val']['latent_smoothness'].append(find_latent_smoothness(all_val_mus))
    metrics['val']['unique_ratio'].append(sum(unique_outputs) / len(unique_outputs))


def train_rnn(
        X,
        y,
        subset_labels,
        device,
        batch_size=32,
        num_epochs=5,
        lr=0.001,
        alpha=0.8,
        lstm_hidden_size=128, # TRY DIFFERENT ENC AND DEC SIZES
        latent_size=64,
        num_lstm_layers=4,
        dropout=0.0,
        use_fc_activations=False
    ):    
    
    
    train_loader, val_loader, test_loader = init_sequential_dataloaders(X, y, batch_size)

    rnn = DoodleGenRNN(
        in_size=4, # 4 features -> dx, dy, dt, p
        hidden_size=lstm_hidden_size, # num features in lstm hidden state
        latent_size=latent_size, # size of vector where new data for generation sampled from
        num_layers=num_lstm_layers, # num of stacked lstm layers
        num_labels=len(subset_labels), # num unique classes in dataset
        dropout=dropout, # chance of neurons to dropout (turn to 0)
        use_fc_activations=use_fc_activations # apply ReLU to non latent space fc layers
    ).to(device)

    reconstruction_criterion = nn.MSELoss() # measure how well did gen and real seqs match (only part of total loss)
    optim = Adam(rnn.parameters(), lr)

    # get shape of sample to give as input to summary
    for batch in train_loader:
        temp_sample_shape = batch[0].shape
        temp_seq_len = batch[1][0].item()
        temp_labels = batch[2]
        break

    model_summary = summary(rnn, input_size=temp_sample_shape, col_names=["input_size", "output_size", "num_params", "trainable"], verbose=0, seq_len=temp_seq_len, labels=temp_labels)
    print(model_summary)

    # logging hyperparams
    hyperparams = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        'num_labels': len(subset_labels),
        'num_samples_per_label': torch.unique(torch.tensor(y), return_counts=True)[1][0].item(),
        "num_total_samples": len(y),
        "lstm_hidden_size": lstm_hidden_size,
        "latent_size": latent_size,
        "num_lstm_layers": num_lstm_layers,
        "kl_div_coeff": alpha,
        "dropout": dropout,
        "use_fc_activations": use_fc_activations
    }
    
    # init metrics dict    
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'hyperparams': hyperparams,
        "device": str(device)
    }

    # train/val loop
    for epoch in range(num_epochs):
        train(
            epoch,
            num_epochs,
            train_loader,
            rnn,
            optim,
            reconstruction_criterion,
            alpha,
            device,
            metrics
        )

        rnn.eval()
        validate(
            epoch,
            num_epochs,
            val_loader,
            rnn,
            reconstruction_criterion,
            alpha, # scale how important kl_div loss is
            device,
            metrics
        )

    # log metrics, generate plot, log model summary, save model
    cur_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = "output/model_metrics/"
    plot_generator_metrics(metrics, cur_time, log_dir)
    log_metrics(metrics, cur_time, log_dir)

    model_fp = "output/models/"
    model_fn = f"DoodleGenRNN_{cur_time}"
    os.makedirs(model_fp, exist_ok=True)
    torch.save(rnn.state_dict(), model_fp + model_fn)
    with open(model_fp + model_fn, 'w') as model_summary_file:
        model_summary_file.write(str(model_summary))
    