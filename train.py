import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import BEAR
import utils

def train_PQL_BEAR(state_dim, action_dim, max_action, device, args):
    print("Training BEARState\n")
    log_name = f"{args.dataset}_{args.seed}"
    # Initialize policy
    policy = BEAR.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
                       version=args.version,
                       lambda_=0.0,
                       threshold=0.05,
                       mode=args.mode,
                       num_samples_match=args.num_samples_match,
                       mmd_sigma=args.mmd_sigma,
                       lagrange_thresh=args.lagrange_thresh,
                       use_kl=(True if args.distance_type == "KL" else False),
                       use_ensemble=(False if args.use_ensemble_variance == "False" else True),
                       kernel_type=args.kernel_type,
                       use_state_vae=True, actor_lr=args.actor_lr, beta=args.beta,
                       n_action=args.n_action, n_action_execute=args.n_action_execute,
                       backup=args.backup, ql_noise=args.ql_noise, vmin=args.vmin
                       )

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{args.dataset}", args.load_buffer_size, bootstrap_dim=4)

    evaluations = []
    training_iters = 0

    while training_iters < args.max_vae_trainstep:
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}")
        print("VAE loss", vae_loss)
        training_iters += args.eval_freq

    if args.automatic_beta:  # args.automatic_beta:
        test_loss = policy.test_vae(replay_buffer, batch_size=100000)
        beta = np.percentile(test_loss, args.beta_percentile)
        policy.beta = beta
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_cpercentile{args.beta_percentile}"
        print("Test vae",args.beta_percentile,"percentile:", beta)
    else:
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_beta{str(args.beta)}"

    if args.backup == "QL":
        hp_setting += f"_ql{args.ql_noise}"

    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/PQL_BEAR_{hp_setting}_{log_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


def train_PQL_BCQ(state_dim, action_dim, max_state, max_action, device, args):
    # For saving files
    log_name = f"{args.dataset}_{args.seed}"

    print("=== Start Train ===\n")
    print("Args:\n",args)

    # Initialize policy
    policy = BCQ.PQL_BCQ(state_dim, action_dim, max_state, max_action, device, args.discount, args.tau, args.lmbda, args.phi,
                         n_action=args.n_action, n_action_execute=args.n_action_execute,
                         backup=args.backup, ql_noise=args.ql_noise,
                         actor_lr=args.actor_lr, beta=args.beta, vmin=args.vmin)

    # Load buffer
    # replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    # replay_buffer.load(f"./buffers/{buffer_name}", args.load_buffer_size)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{args.dataset}", args.load_buffer_size)

    evaluations = []
    filter_scores = []

    training_iters = 0
    while training_iters < args.max_vae_trainstep:
        vae_loss = policy.train_vae(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}. State VAE loss: {vae_loss:.3f}.")
        training_iters += args.eval_freq

    if args.automatic_beta:  # args.automatic_beta:
        test_loss = policy.test_vae(replay_buffer, batch_size=100000)
        beta = np.percentile(test_loss, args.beta_percentile)
        policy.beta = beta
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_cpercentile{args.beta_percentile}"
        print("Test vae",args.beta_percentile,"percentile:", beta)
    else:
        hp_setting = f"N{args.load_buffer_size}_phi{args.phi}_n{args.n_action}_beta{str(args.beta)}"

    if args.backup == "QL":
        hp_setting += f"_ql{args.ql_noise}"

    # Start training
    print("Log files at:", f"./results/BCQState_{hp_setting}_{log_name}")
    training_iters = 0
    while training_iters < args.max_timesteps:
        policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        evaluations.append(eval_policy(policy, args.env, args.seed, eval_episodes=20))
        np.save(f"./results/PQL_{hp_setting}_{log_name}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v2")  # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--dataset", default="d4rl-hopper-medium-v0")  # D4RL dataset name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--max_vae_trainstep", default=0, type=int)

    # BCQ parameter
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.1, type=float)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--load_buffer_size", default=1000000, type=int)  # number of samples to load into the buffer
    parser.add_argument("--actor_lr", default=1e-3, type=float) # learning rate of actor
    parser.add_argument("--n_action", default=100, type=int) # number of sampling action for policy (in backup)
    parser.add_argument("--n_action_execute", default=100, type=int) # number of sampling action for policy (in execution)

    # BEAR parameter
    parser.add_argument("--bear", action="store_true")  # If true, use BEAR
    parser.add_argument("--version", default='0',
                        type=str)  # Basically whether to do min(Q), max(Q), mean(Q)
    parser.add_argument('--mode', default='hardcoded', #hardcoded
                        type=str)  # Whether to do automatic lagrange dual descent or manually tune coefficient of the MMD loss (prefered "auto")
    parser.add_argument('--num_samples_match', default=5, type=int)  # number of samples to do matching in MMD
    parser.add_argument('--mmd_sigma', default=20.0, type=float)  # The bandwidth of the MMD kernel parameter default 10
    parser.add_argument('--kernel_type', default='laplacian',
                        type=str)  # kernel type for MMD ("laplacian" or "gaussian")
    parser.add_argument('--lagrange_thresh', default=10.0,
                        type=float)  # What is the threshold for the lagrange multiplier
    parser.add_argument('--distance_type', default="MMD", type=str)  # Distance type ("KL" or "MMD")
    parser.add_argument('--use_ensemble_variance', default='False', type=str)  # Whether to use ensemble variance or not

    # Our parameter
    parser.add_argument("--backup", type=str, default="QL") # "QL": q learning (Q-max) back up, "AC": actor-critic backup
    parser.add_argument("--ql_noise", type=float, default=0.15) # Noise of next action in QL
    parser.add_argument("--automatic_beta", type=bool, default=True)  # If true, use percentile for b (beta is the b in paper)
    parser.add_argument("--beta_percentile", type=float, default=2.0)  # Use x-Percentile as the value of b
    parser.add_argument("--beta", default=-0.4, type=float)  # hardcoded b, only effective when automatic_beta = False
    parser.add_argument("--vmin", default=0, type=float) # min value of the environment. Empirically I set it to be the min of 1000 random rollout.

    args = parser.parse_args()

    print("---------------------------------------")
    if args.bear:
            print(f"Setting: Training PQL-BEAR, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training PQL-BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_state = float(env.observation_space.high[0])
    if max_state == np.inf:
        max_state = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bear:
        train_PQL_BEAR(state_dim, action_dim, max_action, device, args)
    else:
        train_PQL_BCQ(state_dim, action_dim, max_state, max_action, device, args)
