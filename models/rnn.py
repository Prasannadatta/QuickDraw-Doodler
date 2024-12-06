import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import os
import inspect

from models.mdn import MDN
from utils.process_data import init_sequential_dataloaders
from utils.metrics_visualize import plot_generator_metrics, log_metrics
    
class DoodleGenRNN(nn.Module):
    def __init__(
            self,
            in_size,
            enc_hidden_size,
            dec_hidden_size,
            attention_size,
            latent_size,
            num_lstm_layers,
            num_labels,
            num_mdn_modes,
            dropout,
            decoder_activations,
            subset_labels
        ):

        super(DoodleGenRNN, self).__init__()

        self.mdn = MDN(num_mdn_modes)

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.decoder_activations = decoder_activations
        self.attention_size = attention_size
        self.num_mdn_modes = num_mdn_modes
        self.subset_labels = subset_labels
        self.latent_size = latent_size

        # encode sequential stroke data with NOT bidirectional LSTM (measuring temporal dynamics)
        # final hidden state of lstm defines the latent space
        self.lstm_encoder = nn.LSTM(
            input_size=in_size,
            hidden_size=enc_hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            batch_first=True, # data is loaded with batch dim first (b, seq_len, points)
            bidirectional=False
        )

        # latent space which will be pulled from for generation
        # latent space -> compressed representation of input data to lower dimensional space
        # like convolutions grab most important features, this latent space will learn the same
        # essentially hidden lstm to latent space, but with reparameterizing to sample from gaussian dist of mean and var of hidden
        self.hidden_to_mu_fc = nn.Linear(enc_hidden_size, latent_size) # * 2 for bidirectionality
        self.hidden_to_logvar_fc = nn.Linear(enc_hidden_size, latent_size)

        # embedding for the labels
        # maps discrete labels (idxs for sketch labels e.g. 1 for cat, 2 for tree),
        # to learned cts dense vectors e.g. 1 -> [0.2, 0.9, 0.34,...] to size of latent space (latent_size)
        # allows model to learn relationship between label idxs and the input data
        self.label_embedding = nn.Embedding(num_labels, latent_size)

        # decoder LSTM
        self.latent_to_hidden_fc_h = nn.Linear(latent_size*2, dec_hidden_size)
        self.latent_to_hidden_fc_c = nn.Linear(latent_size*2, dec_hidden_size)
        self.lstm_decoder = nn.LSTM(
            input_size=in_size,
            hidden_size=dec_hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        # different input sizes for with and without attention mechanisms
        # output size is for Gaussian mixture density components
        #  -> 8 gaussian params (mu_dx, mu_dy, mu_dt, sigma_dx, sigma_dy. sigma_dt, rho (correlation), pi, (mixing coefficient))
        # num_mdn_modes -> how many modes the Mixture Density Network will predict (default 20)
        # +3 -> pen states 0, 1, 2
        if attention_size > 0:
            self.hidden_to_output_fc = nn.Linear(enc_hidden_size + dec_hidden_size, (8 * num_mdn_modes) + 3)
        else:
            self.hidden_to_output_fc = nn.Linear(dec_hidden_size, (8 * num_mdn_modes) + 3)

        self.dropout_fc = nn.Dropout(dropout) # dropout layer

        # attention layers
        if attention_size > 0:
            self.attn_enc_fc = nn.Linear(enc_hidden_size, attention_size)
            self.attn_dec_fc = nn.Linear(dec_hidden_size, attention_size)
            self.attn_scalar_fc = nn.Linear(attention_size, 1, bias=False)

    def encode(self, x):
        """
        Encode input sequence of strokes into latent space 
        Final hidden state of lstm is the encoded sequence (compressed/summarized input)

        """
        # grab final hidden state of lstm encoder
        encoder_output, (hn, _) = self.lstm_encoder(x)
            
        # since our lstm is bidirectional we grab the forward and backward states
        hidden = hn[-1]

        # mu and logvar aren't the actual mean and log of variance of hidden state,
        # rather they're learned params we will use as the "mean" and log of "variance" to sample z from
        mu = self.hidden_to_mu_fc(hidden) # mean of latent vec (z)
        logvar = self.hidden_to_logvar_fc(hidden) # log of variance of latent vec, use log for numerical stability

        return mu, logvar, encoder_output

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
        epsilon = torch.randn_like(sigma, device=sigma.device) # epsilon ~ N(0,1) -> random sample from normal dis
        z = mu + epsilon * sigma # adds some random var to latent space to increase generation diversity

        return z

    def attention(self, encoder_output, decoder_output):
        # attention compute (expand for broadcasting)
        encoder_output_attn = self.attn_enc_fc(encoder_output).unsqueeze(1)  # (batch_size, 1, seq_len_dec, attn_dim)
        decoder_hidden_attn = self.attn_dec_fc(decoder_output).unsqueeze(2) # (batch_size, seq_len_dec, 1, attn_dim)
        
        # energy
        energy = torch.tanh(decoder_hidden_attn + encoder_output_attn) # (batch_size, seq_len_dec, seq_len_enc, attn_dim)

        # attention scores
        attn_scores = self.attn_scalar_fc(energy).squeeze(-1) # (batch_size, seq_len_dec, seq_len_enc)

        # attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # context vectors
        context = torch.bmm(attn_weights, encoder_output)  # (batch_size, seq_len_dec, enc_hidden_size)
        decode_context = torch.cat((decoder_output, context), dim=2)
        decode_context = self.dropout_fc(decode_context)

        return decode_context

    def decode(self, z, seq_len, labels, encoder_output, inputs=None):
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
        h0 = self.latent_to_hidden_fc_h(z_conditioned).unsqueeze(0)
        c0 = self.latent_to_hidden_fc_c(z_conditioned).unsqueeze(0)
        
        # apply appropriate activation to hidden state
        if self.decoder_activations.lower() != 'none':
            activation = getattr(F, self.decoder_activations.lower())
            h0 = activation(h0)
            c0 = activation(c0)

        # init hidden state from z that is mapped from the above fc layer is repeated across lstm layers
        h0 = h0.repeat(self.lstm_decoder.num_layers, 1, 1)
        c0 = c0.repeat(self.lstm_decoder.num_layers, 1, 1)
        hidden = (h0, c0)

        # if inputs isn't none,
        # it will contain the original input sequence
        # allowing the decoder to learn separately from the encoder's messy outputs.
        # Prepare inputs
        if inputs is None:
            inputs = torch.zeros(batch_size, seq_len, self.lstm_decoder.input_size, device=z.device)

        # decoding start
        # pass entire sequence through decoder
        decoder_output, _ = self.lstm_decoder(inputs, hidden)

        if self.attention_size > 0:
            decode_context = self.attention(encoder_output, decoder_output)
            outputs = self.hidden_to_output_fc(decode_context)
        else:
            outputs = self.hidden_to_output_fc(decoder_output)            

        return outputs

    def forward(self, x, seq_len, labels, inputs=None):
        mu, logvar, encoder_output = self.encode(x) # encode input sequence
        z = self.reparameterize(mu, logvar) # sample latent vector from encoded sequence
        out = self.decode(z, seq_len, labels, encoder_output, inputs) # decode latent vector with label condition

        return out, mu, logvar


def compute_anneal_factor(step, kl_weight_start, kl_decay_rate):
    return 1 - (1 - kl_weight_start) * (kl_decay_rate ** step)

def kl_divergence_loss(mu, logvar, anneal_factor):
    """
    Calculate KL divergence loss with annealing.
    Args:
        mu: Latent mean (batch_size, latent_size).
        logvar: Latent log-variance (batch_size, latent_size).
        anneal_factor: Weight for the KL loss.
    Returns:
        L_KL: Scalar KL divergence loss.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return anneal_factor * kl / mu.size(0)

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
        anneal_factor,
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
        
        # teacher forcing, give decoder sequence and correct next step
        decoder_inputs = Xbatch[:, :-1, :] # remove last item in sequence
        decoder_target = Xbatch[:, 1:, :] # remove first item
        seq_lens = seq_lens - 1

        # forward
        outputs, mu, logvar = rnn.forward(Xbatch, decoder_inputs.size(1), ybatch, inputs=decoder_inputs)
        all_train_mus.append(mu.detach().cpu()) # store for latent smoothness metric later

        # loss computation
        # mask to only find loss for valid sequence points (not padded points)
        max_seq_len = decoder_inputs.size(1)
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

        # Loss calculations using Gaussian Mixture Density Network for reconstruction and kl divergence loss with annealing
        rnn.mdn.get_mixture_coeff(outputs)
        rec_loss = rnn.mdn.reconstruction_loss(decoder_target, mask)

        #rec_loss = reconstruction_criterion(outputs[mask], decoder_target[mask]) / (seq_lens.sum().item())
        kl_div_loss = kl_divergence_loss(mu, logvar, anneal_factor) / Xbatch.size(0) # how well does generated dis match real data dist
        loss = rec_loss + kl_div_loss # total loss
        
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
        anneal_factor,
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
            
            decoder_inputs = Xbatch[:, :-1, :] # remove last item in sequence
            decoder_target = Xbatch[:, 1:, :] # remove first item
            seq_lens = seq_lens - 1

            # forward
            outputs, mu, logvar = rnn.forward(Xbatch, decoder_inputs.size(1), ybatch, inputs=decoder_inputs)
            all_val_mus.append(mu.detach().cpu()) # store for latent smoothness metric later

            # mask to only find loss for valid sequence points (not padded points)
            max_seq_len = decoder_inputs.size(1)
            mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

            # Loss calculations using Gaussian Mixture Density Network for reconstruction and kl divergence loss with annealing
            rnn.mdn.get_mixture_coeff(outputs)
            rec_loss = rnn.mdn.reconstruction_loss(decoder_target, mask)

            kl_div_loss = kl_divergence_loss(mu, logvar, anneal_factor) / Xbatch.size(0) # how well does generated dis match real data dist
            loss = rec_loss + kl_div_loss # total loss
            
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
        rnn_config,
    ):    
    
    # get model's params and filter config file for them
    rnn_config.update({'num_labels': len(subset_labels)})
    rnn_config.update({'subset_labels': subset_labels})
    rnn_signature = inspect.signature(DoodleGenRNN.__init__)
    rnn_params = {k: v for k, v in rnn_config.items() if k in rnn_signature.parameters}

    # prepare dataloaders
    train_loader, val_loader, test_loader = init_sequential_dataloaders(X, y, rnn_config['batch_size'])

    rnn = DoodleGenRNN(**rnn_params).to(device)

    reconstruction_criterion = nn.MSELoss() # measure how well did gen and real seqs match (only part of total loss)
    optim = Adam(rnn.parameters(), rnn_config['learning_rate'])

    # get shape of sample to give as input to summary
    for batch in train_loader:
        temp_sample_shape = batch[0].shape
        temp_seq_len = batch[1][0].item()
        temp_labels = batch[2]
        break

    model_summary = summary(rnn, input_size=temp_sample_shape, col_names=["input_size", "output_size", "num_params", "trainable"], verbose=0, seq_len=temp_seq_len, labels=temp_labels)
    print(model_summary)
    
    # init metrics dict    
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'hyperparams': rnn_params,
        "device": str(device)
    }

    # train/val loop
    global_step = 0
    start_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}" # for saving model
    for epoch in range(rnn_config['num_epochs']):
        anneal_factor = compute_anneal_factor(global_step, rnn_config['kl_weight_start'], rnn_config['kl_decay_rate'])
        
        train(
            epoch,
            rnn_config['num_epochs'],
            train_loader,
            rnn,
            optim,
            anneal_factor,
            device,
            metrics
        )

        rnn.eval()
        validate(
            epoch,
            rnn_config['num_epochs'],
            val_loader,
            rnn,
            anneal_factor,
            device,
            metrics
        )
        global_step += len(train_loader)

        # each epoch log metrics, generate plot, log model summary, save model
        model_fp = "output/trained_models/"
        model_fn = f"DoodleGenRNN_epoch{epoch+1}_{start_time}"
        os.makedirs(model_fp, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': rnn.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_loss': metrics['train']['total_loss'][epoch],
            'val_loss': metrics['val']['total_loss'][epoch],
            'hyperparams': rnn_params
        }, model_fp + model_fn + '.pt')

        with open(model_fp + model_fn + '.log', 'w', encoding='utf-8') as model_summary_file:
            model_summary_file.write(str(model_summary))

        log_dir = "output/model_metrics/"
        plot_generator_metrics(metrics, start_time, epoch+1, log_dir)
        log_metrics(metrics, start_time, epoch+1, log_dir)
    


