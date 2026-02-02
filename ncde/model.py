import torch
import torchcde
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Try to import RnCLoss, handle case where it's not available
try:
    # Add Rank-N-Contrast to path if RNC_PATH environment variable is set
    rnc_path = os.environ.get('RNC_PATH')
    if rnc_path:
        sys.path.insert(0, rnc_path)
    from loss import RnCLoss
    RNC_AVAILABLE = True
except ImportError:
    RNC_AVAILABLE = False
    RnCLoss = None


# about stability of CDE: https://github.com/rtqichen/torchdiffeq/issues/57
# https://github.com/patrick-kidger/NeuralCDE/issues/2

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, layer_num=2):
        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        # a feed forward network
        # input is (batch, hidden_channels)
        # output is (batch, hidden_channels*input_channels)
        if layer_num == 2:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels*input_channels),
                torch.nn.Tanh()
            )
        elif layer_num == 3:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels*input_channels),
                torch.nn.Tanh()
            )
        elif layer_num == 4:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels*input_channels),
                torch.nn.Tanh()
            )
        else:
            raise ValueError('Invalid layer_num')
    def forward(self, t, z):
        batch_dims = z.shape[:-1]
        return self.network(z).tanh().view(*batch_dims, self.hidden_channels, self.input_channels)


class DSMHead(torch.nn.Module):
    def _init_dsm_layers(self, lastdim):
        if self.dist in ['Weibull']:
            self.act = nn.SELU()
            self.shape = nn.Parameter(-torch.ones(self.k))
            self.scale = nn.Parameter(-torch.ones(self.k))
        elif self.dist in ['LogNormal']:
            self.act = nn.Tanh()
            self.shape = nn.Parameter(torch.ones(self.k))
            self.scale = nn.Parameter(torch.ones(self.k))
        self.gate = nn.Linear(lastdim, self.k, bias=False)
        self.scaleg = nn.Linear(lastdim, self.k, bias=True)
        self.shapeg = nn.Linear(lastdim, self.k, bias=True)
    
    def __init__(self, input_dim, k, dist='Weibull', temp=1000.0):
        super(DSMHead, self).__init__()
        self.k = k
        self.dist = dist
        self.temp = temp
        self._init_dsm_layers(input_dim)
        
    def forward(self, x):
        x = nn.ReLU6()(x)
        batch_size = x.size(0)
        logits = self.gate(x) / self.temp
        shape = self.act(self.shapeg(x)) + self.shape.expand(batch_size, -1)
        scale = self.act(self.scaleg(x)) + self.scale.expand(batch_size, -1)
        return shape, scale, logits
    
    
    def get_shape_scale(self):
        return self.shape, self.scale

    def _get_weibull_logprob(self, shape, scale, t):
        # log P(T=t|shape, scale)
        t = t.unsqueeze(1).expand(-1, self.k)
        k = shape
        b = scale
        s = -torch.pow(torch.exp(b) * t, torch.exp(k))
        f = k + b + s + (torch.exp(k) - 1) * (torch.log(t) + b)
        return f, s # f is the log pdf, s is the log survival function

    def _weibull_loss(self, shape, scale, logits, t, e):
        logcondprob, logcondcum = self._get_weibull_logprob(shape, scale, t)
        prob_mixture = F.softmax(logits, dim=1)
        # maximize the lower bound
        logprob = torch.sum(logcondprob * prob_mixture, dim=1)  # f
        logcum = torch.sum(logcondcum * prob_mixture, dim=1)    # s
        
        loss = -torch.mean(e * logprob + (1 - e) * logcum)
        return loss
    
    
    def _unconditional_weibull_loss(self, t, e):
        k = self.shape.expand(t.size(0), -1)
        b = self.scale.expand(t.size(0), -1)
        t = t.unsqueeze(1).expand(-1, self.k)
        s = -torch.pow(torch.exp(b) * t, torch.exp(k))
        f = k + b + s + (torch.exp(k) - 1) * (torch.log(t) + b)
        e = e.unsqueeze(1).expand(-1, self.k)
        loss = -torch.mean(e * f + (1 - e) * s)
        return loss
    
    def _unconditional_lognormal_loss(self, t, e):
        mu = self.shape
        sigma = torch.exp(self.scale)
        t1 = t.unsqueeze(1).expand(-1, self.k)
        logcondprob = -torch.log(t1 * sigma + 1e-6) - 0.5 * torch.pow((torch.log(t1) - mu) / sigma, 2) - .5 * np.log(2 * np.pi)
        logcondcum = -.5 - .5 * torch.erf((torch.log(t1) - mu) / (sigma * 2 ** 0.5))
        logprob = torch.sum(logcondprob, dim=1)
        logcum = torch.sum(logcondcum, dim=1)
        loss = -torch.mean(e * logprob + (1 - e) * logcum)
        return loss
    
    def compute_unconditional_loss(self, t, e):
        if self.dist in ['Weibull']:
            return self._unconditional_weibull_loss(t, e)
        elif self.dist in ['LogNormal']:
            return self._unconditional_lognormal_loss(t, e)
        else:
            raise ValueError('Invalid distribution')
    
    
    def _lognormal_loss(self, shape, scale, logits, t, e):
        mu = shape
        sigma = torch.exp(scale)
        t1 = t.unsqueeze(1).expand(-1, self.k)
        logcondprob = -torch.log(t1 * sigma + 1e-6) - 0.5 * torch.pow((torch.log(t1) - mu) / sigma, 2) - .5 * np.log(2 * np.pi)
        logcondcum = -.5 - .5 * torch.erf((torch.log(t1) - mu) / (sigma * 2 ** 0.5))
        prob_mixture = F.softmax(logits, dim=1)
        logprob = torch.sum(logcondprob * prob_mixture, dim=1)
        logcum = torch.sum(logcondcum * prob_mixture, dim=1)
        loss = -torch.mean(e * logprob + (1 - e) * logcum)
        return loss

    def _get_lognormal_logcum(self, shape, scale, t):
        mu = shape
        sigma = torch.exp(scale)
        t = t.unsqueeze(1).expand(-1, self.k)
        return -.5 - .5 * torch.erf((torch.log(t) - mu) / (sigma * 2 ** 0.5))

    def compute_loss(self, x, t, e):
        shape, scale, logits = self(x)  # (batch, k)
        if self.dist in ['Weibull']:
            return self._weibull_loss(shape, scale, logits, t, e)
        elif self.dist in ['LogNormal']:
            return self._lognormal_loss(shape, scale, logits, t, e)
        else:
            raise ValueError('Invalid distribution')
    
    def _weibull_cdf(self, x, t):
        shape, scale, logits = self(x)
        logits = F.log_softmax(logits, dim=1)
        k = shape
        b = scale
        t = t.unsqueeze(1).expand(-1, self.k)
        s = -torch.pow(torch.exp(b) * t, torch.exp(k))
        log_cdf = s + logits
        log_cdf = torch.logsumexp(log_cdf, dim=1)
        return torch.exp(log_cdf)
    
    def predict_surv(self, x, t):
        # P(T>t|X)
        shape, scale, logits = self(x)
        if self.dist in ['Weibull']:
            return self._weibull_cdf(x, t)  # log P(T>t|X, k)
        elif self.dist in ['LogNormal']:
            logcondcum = self._get_lognormal_logcum(shape, scale, t)    # log P(T>t|X, k)
        log_prob_mixture = F.log_softmax(logits, dim=1) # log P(k|X)
        # P(T>t|X) = \sum_k exp(log(P(T>t|X, k)))P(k|X)
        log_prob = torch.logsumexp(logcondcum + log_prob_mixture, dim=1)
        return torch.exp(log_prob)
    
    def predict_risk(self, x, t):
        return 1 - self.predict_surv(x, t)
    
        

class CoxHead(torch.nn.Module):
    # cox proportional hazard model
    def __init__(self, input_dim, nonlinear=False):
        super(CoxHead, self).__init__()
        if not nonlinear:
            self.linear = torch.nn.Linear(input_dim, 1)
        else:
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            )
            
        self.baseline_hazard_flag = False

    def forward(self, x):
        return self.linear(x).squeeze()

    def compute_loss(self, x, t, e):
        # x: (batch, input_dim)
        # t: (batch,)
        # e: (batch,)
        sorted_indices = torch.argsort(t, descending=True)
        x = x[sorted_indices]
        t = t[sorted_indices]
        e = e[sorted_indices]
        risk = self(x)
        
        risk_set_mask = t.unsqueeze(1) <= t.unsqueeze(0)    # originally >=
        adjusted_risk_set_mask = torch.where(risk_set_mask, 0, float('-inf'))
        # expand risk to a matrix
        expanded_risk = risk.unsqueeze(0).expand(risk.size(0), -1)
        log_risk_sums = torch.logsumexp(expanded_risk + adjusted_risk_set_mask, dim=1)
        log_likelihood = risk - log_risk_sums
        loss = -torch.sum(log_likelihood * e)
        
        return loss
    
    def pairwise_ranking(self, x, t, e):
        # pairwise ranking loss with smoothness
        risk = self(x)
        risk_set_mask = t.unsqueeze(1) <= t.unsqueeze(0)
        risk_diff = risk.unsqueeze(1) - risk.unsqueeze(0)
        soft_risk_diff = torch.sigmoid(-risk_diff)
        loss = torch.sum(soft_risk_diff * risk_set_mask.float(), dim=1)
        # only consider events
        loss = loss * e
        return torch.sum(loss)
    
    
    
    def fit_baseline_cum_hazard(self, x, t, e):
        # fit H0 with Breslow estimator
        risk = self(x)
        sorted_indices = torch.argsort(t, descending=False)
        risk = risk[sorted_indices]
        t = t[sorted_indices]
        e = e[sorted_indices]
        unique_times, indices = torch.unique(t, return_inverse=True)
        hazard_ratio = torch.exp(risk)
        cum_hazard = torch.zeros_like(unique_times, dtype=torch.float)
        for i, time in enumerate(unique_times):
            event_mask = (t == time) & e
            sum_hazard_ratios_at_risk = torch.sum(hazard_ratio[indices >= i])
            cum_hazard[i] = torch.sum(event_mask.float()) / sum_hazard_ratios_at_risk
        cum_hazard = torch.cumsum(cum_hazard, dim=0)
        self.unique_times = unique_times
        self.baseline_cum_hazard = cum_hazard
        
        self.baseline_hazard_flag = True
            

    
    
    def predict_surv(self, x, t):
        assert self.baseline_hazard_flag, 'Please fit the baseline hazard first'
        # exp(-H0(t) * exp(x))
        interp_cum_hazard = F.interp(t, self.unique_times, self.baseline_cum_hazard)
        risk = self(x)
        return torch.exp(-interp_cum_hazard * torch.exp(risk))
    
    def predict_risk(self, x, t):
        assert self.baseline_hazard_flag, 'Please fit the baseline hazard first'
        return 1 - self.predict_surv(x, t)
    
    def predict_relative_surv(self, x, t):
        return torch.exp(-self(x))

    def predict_relative_risk(self, x, t):
        return 1 - self.predict_relative_surv(x, t)



class SoftSilhouetteLoss(nn.Module):
    def __init__(self, num_clusters, temperature=1.0):
        super(SoftSilhouetteLoss, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature

    def forward(self, hidden_states):
        # Compute soft assignments using a softmax function
        centroids = hidden_states[torch.randperm(hidden_states.size(0))[:self.num_clusters]]
        logits = -torch.cdist(hidden_states, centroids, p=2)
        soft_assignments = F.softmax(logits / self.temperature, dim=1)

        # Compute pairwise distances
        pairwise_distances = torch.cdist(hidden_states, hidden_states, p=2)

        # Compute soft a(i) and b(i)
        a = torch.zeros(hidden_states.size(0), device=hidden_states.device)
        b = torch.zeros(hidden_states.size(0), device=hidden_states.device)
        
        for i in range(hidden_states.size(0)):
            # Intra-cluster distances
            intra_cluster_distances = pairwise_distances[i] * soft_assignments[:, soft_assignments[i].argmax()]
            a[i] = torch.sum(intra_cluster_distances) / (torch.sum(soft_assignments[:, soft_assignments[i].argmax()]) - soft_assignments[i].max())

            # Inter-cluster distances
            inter_cluster_distances = pairwise_distances[i].unsqueeze(0) * (1 - soft_assignments[i]).unsqueeze(1)
            b[i] = torch.min(torch.sum(inter_cluster_distances, dim=1) / torch.sum(1 - soft_assignments[i], dim=0))

        # Calculate silhouette coefficients for each sample
        sil_coeff = (b - a) / torch.max(a, b)
        
        # Handle cases where a and b are both zero
        sil_coeff[torch.isnan(sil_coeff)] = 0

        # Return the mean silhouette coefficient as the loss
        return -torch.mean(sil_coeff)  # Typically, we minimize the negative silhouette coefficient


    
    


class NCDESurv(torch.nn.Module):
    def __init__(self, 
                 input_channels, 
                 hidden_channels, 
                 k=3,
                 dist='Weibull',
                 temp=1000.0,
                 head="cox",
                 layer_num=2,
                 nonlinear=False,
                 pairwise=False,
                 silhouette=False,
                 num_clusters=5,
                 silhouette_weight=.01,
                 contrastive=False,
                 contrastive_weight=1.0,
                 ae=False,
                 vae=False,
                 weight_decay_temp=10,
                 no_ffn=False,
                 **kwargs):
        '''
        input_channels: int, the number of input channels
        hidden_channels: int, dimension of latent space
        k: int, the number of components in the mixture distribution
        dist: str, the distribution to use, 'Weibull' or 'LogNormal'
        temp: float, the temperature for the gumbel softmax
        '''
        super(NCDESurv, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.k = k
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.func = NeuralCDE(input_channels, hidden_channels, layer_num)
        self.pairwise = pairwise
        self.no_ffn = no_ffn
        if head == 'cox':
            self.head = CoxHead(hidden_channels, nonlinear)
        elif head == 'dsm':
            self.head = DSMHead(hidden_channels, k, dist, temp)
        if silhouette:
            print("Silhouette loss is used", silhouette, num_clusters, silhouette_weight)
            self.silhouette = SoftSilhouetteLoss(num_clusters=num_clusters)
            self.silhouette_weight = silhouette_weight
            self.use_silhouette = True
        else:
            self.use_silhouette = False
        if contrastive:
            if not RNC_AVAILABLE:
                raise ImportError("RnCLoss not available. Please set RNC_PATH environment variable to the Rank-N-Contrast directory.")
            print("Contrastive loss is used", contrastive, contrastive_weight)
            self.contrastive = True
            self.criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2', weight_decay_temp=weight_decay_temp)
            self.contrastive_weight = contrastive_weight
            if self.no_ffn:
                self.contrastive_head = torch.nn.Sequential()
            else:
                self.contrastive_head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, hidden_channels)
                )
        else:
            self.contrastive = False
        if ae:
            print("Autoencoder used")
            if not contrastive:
                self.contrastive_head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, hidden_channels)
                )
            self.ae_decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels)
            )
            self.ae = True
        else:
            self.ae = False
        
        if vae:
            print("VAE used for uncertainty quantification")
            if not contrastive:
                self.contrastive_head = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, hidden_channels)
                )
            self.var_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels)
            )
            self.ae_decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, hidden_channels)
            )
            
        

    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
        zT = zt[..., -1, :]
        if self.contrastive or self.ae:
            zT = self.contrastive_head(zT)
        return zT
    
    def compute_loss(self, x, t, e, sofa_label=None, x_length=None, times=None):
        # x: (batch, seq, input_channels)
        # t: (batch,)
        # e: (batch,)
        all_states = self.get_latent_state(x, which_time='all')
        if self.contrastive or self.ae:
            last_state = self.contrastive_head(all_states[..., -1, :])
        else:
            last_state = all_states[..., -1, :]
        cox_loss = self.head.compute_loss(last_state, t, e)
        return_dict = {
            'cox_loss': cox_loss,
            'total_loss': cox_loss
        }
        if self.pairwise:
            pairwise_loss = self.head.pairwise_ranking(last_state, t, e)
            return_dict['pairwise_loss'] = pairwise_loss
            return_dict['total_loss'] = return_dict['total_loss'] + pairwise_loss
        if self.use_silhouette:
            silhouette_loss = self.silhouette(last_state)
            return_dict['silhouette_loss'] = silhouette_loss
            return_dict['total_loss'] = return_dict['total_loss'] + silhouette_loss * self.silhouette_weight
        if self.contrastive:
            if self.no_ffn:              
                contrastive_loss = self.compute_contrastive_loss(all_states, sofa_label, x_length, times)
            else:
                contrastive_loss = self.compute_contrastive_loss(all_states.detach(), sofa_label, x_length, times)
            return_dict['contrastive_loss'] = contrastive_loss
            if self.contrastive_weight > 0 and torch.isfinite(contrastive_loss):
                return_dict['total_loss'] = return_dict['total_loss'] + contrastive_loss * self.contrastive_weight
        if self.ae:
            if self.no_ffn:
                ae_loss = self.compute_ae_loss(all_states)
            else:
                ae_loss = self.compute_ae_loss(all_states.detach())
            return_dict['ae_loss'] = ae_loss
            return_dict['total_loss'] = return_dict['total_loss'] + .5 * ae_loss
            
        return return_dict

    def compute_ae_loss(self, z):
        z1 = self.contrastive_head(z)
        reconst_z = self.ae_decoder(z1)
        # mse loss
        mse_criterion = nn.MSELoss()
        loss = mse_criterion(reconst_z, z)
        return loss

    def compute_contrastive_loss(self, z, sofa_label, x_length, times=None):
        # z.shape = (batch, seq, hidden_channels)
        # sofa_label.shape = (all_length, 2)
        # x_length.shape = (batch,) and the sum of x_length is equal to all_length
        
        # for each sample i, z[i][:x_length[i]] is the latent state of the i-th sample
        # get the latent states for all samples using the x_length
        z = self.contrastive_head(z)
        latent_states = [z[i][:x_length[i]] for i in range(z.size(0))]
        latent_states = torch.cat(latent_states, dim=0) # (all_length, hidden_channels)
        # make it (all_length, 2, hidden_channels) by repeating the latent_states
        latent_states = latent_states.unsqueeze(1).expand(-1, 2, -1)
        loss = self.criterion(latent_states, sofa_label, times)
        return loss
    
    def compute_only_contrastive_loss(self, x, sofa_label, x_length):
        z = self.get_latent_state(x, which_time='all')
        loss = self.compute_contrastive_loss(z, sofa_label, x_length)
        return {
            'contrastive_loss': loss,
            'total_loss': loss
        }
        

    def predict_surv(self, x, t, relative=True):
        if relative:
            return self.head.predict_relative_surv(self(x), t)
        else:
            return self.head.predict_surv(self(x), t)
    
    def predict_risk(self, x, t, relative=True):
        if relative:
            return self.head.predict_relative_risk(self(x), t)
        else:
            return self.head.predict_risk(self(x), t)
    
    def get_last_latent_state(self, x):
        return self(x)
    
    def get_latent_trajectory(self, x):
        X = torchcde.CubicSpline(x)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)

        return zt
    
    def get_latent_state(self, x, which_time='last', t=None):
        X = torchcde.CubicSpline(x)
        t = torch.FloatTensor(list(range(int(X.interval[0]), int(X.interval[-1] + 1)))).to(x.device) if t is None else t
        assert t[0] == X.interval[0], 'The start time is not the same'
        assert t[-1] == X.interval[-1], 'The end time is not the same'
        X0 = X.evaluate(t[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=t)

        if which_time == 'last':
            return zt[..., -1, :]
        elif which_time == 'first':
            return zt[..., 0, :]
        else:
            return zt   # (batch, len(t), latent_dim)
        
    def get_one_latent_trajectory(self, x, t):
        # x is the coeff, t is the time steps of each observation (interval_id)
        X = torchcde.CubicSpline(x)
        X0 = X.evaluate(t[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=t).squeeze()
        return zt
