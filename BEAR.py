import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch.distributions as td

from BCQ import VAE_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Actor used in BCQ"""

    def __init__(self, state_dim, action_dim, max_action, threshold=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


def add_gaussian_noise(actions, max_action, std):
    return (
        actions
        + max_action * std * torch.randn_like(actions)
    ).clamp(-max_action, max_action)


class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""

    def __init__(self, state_dim, action_dim, max_action, ):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        #mean_a = mean_a.clamp(-20,20)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device)
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        #mean_a = mean_a.clamp(-20, 20)
        log_std_a = self.log_std(a)

        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) + \
            std_a.unsqueeze(1) * torch.FloatTensor(
                np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        #mean_a = mean_a.clamp(-20, 20)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action ** 2).clamp(min=1e-6).log().sum(-1)
        return log_pis


class Critic(nn.Module):
    """Regular critic used in off-policy RL"""

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class EnsembleCritic(nn.Module):
    """ Critic which does have a network of 4 Q-functions"""

    def __init__(self, num_qs, state_dim, action_dim):
        super(EnsembleCritic, self).__init__()

        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        # self.l7 = nn.Linear(state_dim + action_dim, 400)
        # self.l8 = nn.Linear(400, 300)
        # self.l9 = nn.Linear(300, 1)

        # self.l10 = nn.Linear(state_dim + action_dim, 400)
        # self.l11 = nn.Linear(400, 300)
        # self.l12 = nn.Linear(300, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []

        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0), ], 0)  # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def q_all(self, state, action, with_var=False):
        all_qs = []

        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0), ], 0)  # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""

    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device)

        u = self.decode(state, z)

        return u, mean, std

    def decode_softplus(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5,
                                                                                                                  0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))

    def decode(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5,
                                                                                                                  0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_bc(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_bc_test(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25,
                                                                                                                  0.25)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(
                device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)


class BEAR(object):
    def __init__(self, num_qs, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=True, version=0,
                 lambda_=0.4,
                 threshold=0.05, mode='auto', num_samples_match=10, mmd_sigma=10.0, actor_lr=1e-3,
                 lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='laplacian', use_state_filter=False,
                 n_action=100, n_action_execute=100, backup="QL", ql_noise=0.0,
                 beta=0.0, vmin = 0):
        latent_dim = action_dim * 2
        self.actor = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.delta_conf = delta_conf
        self.use_bootstrap = use_bootstrap
        self.version = version
        self._lambda = lambda_
        self.threshold = threshold
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_kl = use_kl
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type

        # state filter
        self.use_state_filter = use_state_filter
        self.beta = beta
        self.n_action = n_action
        self.n_action_execute = n_action_execute
        self.backup = backup
        self.ql_noise = ql_noise
        self.vmin = vmin

        if self.mode == 'auto':
            # Use lagrange multipliers on the constraint if set to auto mode
            # for the purpose of maintaing support matching at all times
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2, ], lr=1e-3)

        if self.use_state_filter:
            self.vae2 = VAE_state(state_dim, state_dim * 2, None, device).to(device)
            self.vae2_optimizer = torch.optim.Adam(self.vae2.parameters())

        self.epoch = 0

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def kl_loss(self, samples1, state, sigma=0.2):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)

    def entropy_loss(self, samples1, state, sigma=0.2):
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        # print (samples1_log_prob.min(), samples1_log_prob.max())
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return (samples1_prob).mean(1)

    def select_action(self, state):
        """When running the actor, we just select action based on the max of the Q-function computed over
            samples from the policy -- which biases things to support."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(self.n_action_execute, 1).to(device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def train_vae(self, replay_buffer, iterations, batch_size=100):
        scores = []

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            recon, mean, std = self.vae2(state)
            recon_loss = F.mse_loss(recon, state)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            scores.append(vae_loss.item())

            self.vae2_optimizer.zero_grad()
            vae_loss.backward()
            self.vae2_optimizer.step()

            recon, mean, std = self.vae2(next_state)
            recon_loss = F.mse_loss(recon, next_state)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            scores.append(vae_loss.item())

            self.vae2_optimizer.zero_grad()
            vae_loss.backward()
            self.vae2_optimizer.step()
        return np.mean(scores)

    def test_vae(self, replay_buffer, batch_size=1000):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        recon, mean, std = self.vae2(next_state)
        recon_loss = ((recon - next_state) ** 2).mean(dim=1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        vae_loss = recon_loss + 0.5 * KL_loss

        return -vae_loss.detach().cpu().numpy()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            state, action, next_state, reward, done, mask = replay_buffer.sample(batch_size, use_bootstrap=True)

            next_state_np = next_state.detach().cpu().numpy()
            state_np = state.detach().cpu().numpy()

            # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training: In this step, we explicitly compute the actions
            with torch.no_grad():
                # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
                state_rep = torch.FloatTensor(np.repeat(next_state_np, self.n_action, axis=0)).to(device)

                # Compute value of perturbed actions sampled from the VAE

                if self.backup == "QL":
                    target_Qs = self.critic_target(next_state, add_gaussian_noise(self.vae.decode(next_state),
                                                                                  self.max_action,
                                                                                  self.ql_noise))
                else:
                    target_Qs = self.critic_target(state_rep, self.actor_target(state_rep))

                if self.use_state_filter and self.beta < 0:
                    recon, mean, std = self.vae2(state_rep)
                    score = - ((recon - state_rep) ** 2).mean(dim=1)
                    score += 0.5 * 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
                    score = score.reshape(batch_size, -1).mean(dim=1, keepdim=True)
                    score = torch.sigmoid(100 * (score - self.beta))
                else:
                    score = 1

                # Soft Clipped Double Q-learning
                target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0]
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
                target_Q = reward + done * discount * score * target_Q + done * discount * (1 - score) * self.vmin

            current_Qs = self.critic(state, action, with_var=False)
            if self.use_bootstrap:
                critic_loss = (F.mse_loss(current_Qs[0], target_Q, reduction='none') * mask[:, 0:1]).mean() + \
                              (F.mse_loss(current_Qs[1], target_Q, reduction='none') * mask[:, 1:2]).mean()
                # (F.mse_loss(current_Qs[2], target_Q, reduction='none') * mask[:, 2:3]).mean() +\
                # (F.mse_loss(current_Qs[3], target_Q, reduction='none') * mask[:, 3:4]).mean()
            else:
                critic_loss = F.mse_loss(current_Qs[0], target_Q) + F.mse_loss(current_Qs[1],
                                                                               target_Q)  # + F.mse_loss(current_Qs[2], target_Q) + F.mse_loss(current_Qs[3], target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Action Training
            # If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
            num_samples = self.num_samples_match
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_samples)  # num)

            # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
            if self.use_kl:
                mmd_loss = self.kl_loss(raw_sampled_actions, state)
            else:
                if self.kernel_type == 'gaussian':
                    mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
                else:
                    mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            action_divergence = ((sampled_actions - actor_actions) ** 2).sum(-1)
            raw_action_divergence = ((raw_sampled_actions - raw_actor_actions) ** 2).sum(-1)

            # Update through TD3 style
            critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
            critic_qs = self.critic.q_all(
                state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples * state.size(0), state.size(1)),
                actor_actions.permute(1, 0, 2).contiguous().view(num_samples * actor_actions.size(0),
                                                                 actor_actions.size(2)))
            critic_qs = critic_qs.view(self.num_qs, num_samples, actor_actions.size(0), 1)
            critic_qs = critic_qs.mean(1)
            std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)

            if not self.use_ensemble:
                std_q = torch.zeros_like(std_q).to(device)

            if self.version == '0':
                critic_qs = critic_qs.min(0)[0]
            elif self.version == '1':
                critic_qs = critic_qs.max(0)[0]
            elif self.version == '2':
                critic_qs = critic_qs.mean(0)

            # We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
            # Change to 10 because I doubled the epoch length
            if self.epoch >= 10:
                if self.mode == 'auto':
                    actor_loss = (-critic_qs + \
                                  self._lambda * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + \
                                  self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = (-critic_qs + \
                                  self._lambda * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + \
                                  100.0 * mmd_loss).mean()  # This coefficient is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
            else:
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0 * mmd_loss.mean()

            std_loss = self._lambda * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q.detach()

            self.actor_optimizer.zero_grad()
            if self.mode == 'auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()

            # check norm
            # norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            norm = 0
            for p in self.actor.parameters():
                param_norm = p.grad.data.norm(2)
                norm += param_norm.item() ** 2
            norm = norm ** (1. / 2)
            # print("\nIteration:", it,
            #       "\nactor_loss:", actor_loss.item(),
            #       "\ngradient norm:", norm,
            #       "\ncritic_qs.mean():", critic_qs.mean().item(),
            #       "\nstd_q.mean():", std_q.mean().item(),
            #       "\nmmd_loss.mean():", mmd_loss.mean().item(),
            #       "\nactor_actions.max():", actor_actions.max()
            #       )
            if np.isnan(norm):
                print("ValueError: nan in gradients. Ending current train.")
                raise AssertionError
                # self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

            # Threshold for the lagrange multiplier
            thresh = 0.05
            if self.use_kl:
                thresh = -2.0

            if self.mode == 'auto':
                lagrange_loss = (-critic_qs + \
                                 self._lambda * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * (std_q) + \
                                 self.log_lagrange2.exp() * (mmd_loss - thresh)).mean()

                self.lagrange2_opt.zero_grad()
                (-lagrange_loss).backward()
                # self.lagrange1_opt.step()
                self.lagrange2_opt.step()
                self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)

                # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Do all logging here
        self.epoch = self.epoch + 1


def weighted_mse_loss(inputs, target, weights):
    return torch.mean(weights * (inputs - target) ** 2)
