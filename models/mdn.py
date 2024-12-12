import torch
import torch.nn.functional as F

def bivariate_tikhonov_regularizer(scales, corrs, alpha):
    """
    Tikhonov regularization (helps stabilize training and ensures cov mat is invertible):
    Adds a small non-negative constant to the diagonal of the covariance matrix 
    for a bivariate distribution.
    alpha -> 'reg_covar' param in 'model_params.json' determines strength of regularization
    """
    # regularize std devs of the bivariate dist
    reg_scales = torch.sqrt(scales**2 + alpha) # (batch_size, 2)
    
    # update correlation coefficients to account for std devs change
    reg_corrs = corrs * torch.prod(scales, -1) / torch.prod(reg_scales, -1) # (batch_size,)
    
    return reg_scales, reg_corrs

class MDN:
    """
    Mixture Density Network to model output mixture coefficients from DoodleGen Decoder

    WARNING: This class is NOT modular, it is designed solely for the output of the above VAE
    TODO make this shit modular, but idc rn it's 3am and I just want this to WORRKKRKRK
    """
    def __init__(self, num_modes):
        self.num_modes = num_modes

    def set_mixture_coeff(self, output):
        """
        split decoder output into MDN parameters and process them

        Args:
            output: Tensor of shape (batch_size, seq_len, output_dim), decoder outputs.
                output_dim = (6 * num_gmm_modes) + 3
            num_gmm_modes: int, number of mixture components (M).

        Returns:
            - z_pi: Tensor of shape (batch_size, seq_len, num_gmm_modes), mixture weights.
            - z_mu1, z_mu2: Tensors of shape (batch_size, seq_len, num_gmm_modes), Gaussian means for x1, x2.
            - z_sigma1, z_sigma2: Tensors of shape (batch_size, seq_len, num_gmm_modes), Gaussian standard deviations for x1, x2.
            - z_corr: Tensor of shape (batch_size, seq_len, num_gmm_modes), correlations in [-1, 1].
            - z_pen_probs: Tensor of shape (batch_size, seq_len, 3), pen state probabilities.
            - z_pen_logits: Tensor of shape (batch_size, seq_len, 3), raw logits for pen states.
        """
        num_gmm_modes = self.num_modes
        
        # Split output
        # (...) notation means grab the last indexed item from all outer dimensions
        # so line below will grab all the z_pis from all sequence lengths and all items in batch
        self.z_pi = output[..., :num_gmm_modes]  # Mixture weights
        self.z_mu_dx = output[..., num_gmm_modes:2*num_gmm_modes]
        self.z_mu_dy = output[..., 2*num_gmm_modes:3*num_gmm_modes]
        self.z_mu_dt = output[..., 3*num_gmm_modes:4*num_gmm_modes]
        self.z_sigma_dx = output[..., 4*num_gmm_modes:5*num_gmm_modes]
        self.z_sigma_dy = output[..., 5*num_gmm_modes:6*num_gmm_modes]
        self.z_sigma_dt = output[..., 6*num_gmm_modes:7*num_gmm_modes]
        self.z_corr = output[..., 7*num_gmm_modes:8*num_gmm_modes]
        self.z_pen_logits = output[..., 8*num_gmm_modes:8*num_gmm_modes+3]

        # Process parameters
        self.z_pi = F.softmax(self.z_pi, dim=-1)  # Softmax for mixture weights
        self.z_pen = F.softmax(self.z_pen_logits, dim=-1)  # Softmax for pen states
        self.z_sigma_dx = torch.exp(self.z_sigma_dx)  # Exponentiate std deviations
        self.z_sigma_dy = torch.exp(self.z_sigma_dy)
        self.z_sigma_dt = torch.exp(self.z_sigma_dt)
        self.z_corr = torch.tanh(self.z_corr)  # tanh for correlations in [-1, 1]

    def _univariate_normal(self, x):
        # x: (batch_size, seq_len)
        # mu, sigma: (batch_size, seq_len, num_gmm_modes)
        x = x.unsqueeze(-1)  # Expand to match mixture components
        prob = (1 / (self.z_sigma_dt * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((x - self.z_mu_dt) / self.z_sigma_dt) ** 2)
        #print("pt_1d_normal prob min:", prob.min().item(), "max:", prob.max().item())

        return prob

    def _bivariate_normal(self, x1, x2, epsilon=1e-6):
        """
        Compute the probability of (x1, x2) under a bivariate Gaussian distribution.

        Args:
            x1: Tensor of shape (batch_size, seq_len, 1), target x1 values.
            x2: Tensor of shape (batch_size, seq_len, 1), target x2 values.
            mu1: Tensor of shape (batch_size, seq_len, num_mixture), predicted mean of x1.
            mu2: Tensor of shape (batch_size, seq_len, num_mixture), predicted mean of x2.
            s1: Tensor of shape (batch_size, seq_len, num_mixture), predicted std dev of x1.
            s2: Tensor of shape (batch_size, seq_len, num_mixture), predicted std dev of x2.
            rho: Tensor of shape (batch_size, seq_len, num_mixture), predicted correlation coefficients.

        Returns:
            prob: Tensor of shape (batch_size, seq_len, num_mixture), bivariate Gaussian probabilities.
        """
        norm1 = (x1.unsqueeze(-1) - self.z_mu_dx) / self.z_sigma_dx
        norm2 = (x2.unsqueeze(-1) - self.z_mu_dy) / self.z_sigma_dy
        z = (norm1 ** 2 + norm2 ** 2 - 2 * self.z_corr * norm1 * norm2) / (1 - self.z_corr ** 2 + epsilon)
        var1var2 = self.z_sigma_dx*self.z_sigma_dy
        prob = (torch.exp(-z / 2)) / (2 * torch.pi * var1var2 * torch.sqrt(1 - self.z_corr ** 2 + epsilon))
        #print("pt_2d_normal prob min:", prob.min().item(), "max:", prob.max().item())
        return prob

    def reconstruction_loss(self, target, mask, reg_covar=0):
        """
        Compute the reconstruction loss for Sketch-RNN.

        Args:
            target: Tensor of shape (batch_size, seq_len, 5), ground truth strokes:
                [x1, x2, pen_lift, pen_down, pen_end].
            z_pi: Tensor of shape (batch_size, seq_len, num_gmm_modes), mixture weights.
            z_mu1, z_mu2: Tensors of shape (batch_size, seq_len, num_gmm_modes), Gaussian means for x1, x2.
            z_sigma1, z_sigma2: Tensors of shape (batch_size, seq_len, num_gmm_modes), Gaussian standard deviations.
            z_corr: Tensor of shape (batch_size, seq_len, num_gmm_modes), correlations in [-1, 1].
            z_pen_logits: Tensor of shape (batch_size, seq_len, 3), raw logits for pen states.
            mask: Tensor of shape (batch_size, seq_len), binary mask for valid timesteps.

        Returns:
            Tensor: Scalar tensor representing the mean reconstruction loss.
        """
        dx, dy, dt, pen_state = target[..., 0], target[..., 1], target[..., 2], target[..., 3].long()  # Split target

        # tikhonov regularization for bivariate distributions
        if reg_covar > 0.:
            # combine scales
            scales = torch.stack([self.z_sigma_dx, self.z_sigma_dy], dim=-1) 
            reg_scales, self.z_corr = bivariate_tikhonov_regularizer(scales, self.z_corr, reg_covar)
            self.z_sigma_dx, self.z_sigma_dy = reg_scales[..., 0], reg_scales[..., 1]  # unpack scales back into stddevs

        # bivariate distribution probabilities for dx and dy
        spatial_prob = self._bivariate_normal(dx, dy)
        spatial_prob = torch.sum(spatial_prob * self.z_pi, dim=-1) + 1e-6  # Avoid log(0)

        # univariate distribution probabilities for dt
        temporal_prob = self._univariate_normal(dt)
        temporal_prob = torch.sum(temporal_prob * self.z_pi, dim=-1) + 1e-6

        # L_s: stroke reconstruction loss
        # assumed spatial and temporal dimensions are independent
        #L_s = -torch.log(spatial_prob * temporal_prob + 1e-6) * mask
        L_s = - (0.5 * torch.log(spatial_prob + 1e-6) +  0.5 * torch.log(temporal_prob + 1e-6)) * mask

        # L_p: CE for pen state
        pen_loss = F.cross_entropy(self.z_pen_logits.view(-1, 3), pen_state.view(-1), reduction='none')
        L_p = pen_loss.view(pen_state.shape) * mask

        #print(L_s, temporal_prob, temporal_prob)

        # Total reconstruction loss
        return (L_s.sum() + L_p.sum()) / mask.sum()

    def sample_mdn(self, t):
        # sample pen state distribution
        pen_state = torch.multinomial(self.z_pen[0, t], 1).item()

        # sample dx, dy, dt
        mode_idx = torch.multinomial(self.z_pi[0, t], 1).item()

        dx = torch.normal(self.z_mu_dx[0, t, mode_idx], self.z_sigma_dx[0, t, mode_idx]).item()
        dy = torch.normal(self.z_mu_dy[0, t, mode_idx], self.z_sigma_dy[0, t, mode_idx]).item()
        dt = torch.normal(self.z_mu_dt[0, t, mode_idx], self.z_sigma_dt[0, t, mode_idx]).item()

        return dx, dy, dt, pen_state