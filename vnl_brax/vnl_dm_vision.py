# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetahBulletEnv-v0",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="the path to a saved model")
    parser.add_argument("--save_directory", type=str, default="checkpoints_ppo",
                        help="the base directory to save the model")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ProprioNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim):
        super(ProprioNetwork, self).__init__()

        # two layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)
        x = self.linear2(x)
        return x


class CoreNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim):
        super(CoreNetwork, self).__init__()

        # two layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()

        # two layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x

# TODO: add vision encoder
import torch.nn as nn
from torchvision.models import alexnet
class AlexNetEncoder(nn.Module):
    def __init__(self, output_dim=16):
        super().__init__()
        model = alexnet(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # staple a layer onto the end of the encoder
        self.staple = nn.Linear(256 * 6 * 6, output_dim)

    def forward(self, x):
        x = self.encoder(x)  # (batch_size, 256, 6, 6)
        # flatten the output to (batch_size, 256 * 6 * 6)
        x = x.view(x.size(0), -1)
        # x = x.flatten(x, 1)
        x = self.staple(x)
        return x



class VisionAgent(nn.Module):
    def __init__(self, action_space, proprio_obs_dim, vision_obs_dim, proprio_encoding_dim=16, vision_encoding_dim=16,
                 hidden_dim=256, use_vision=True, device='cpu'):
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.proprio_obs_dim = proprio_obs_dim
        self.vision_obs_dim = vision_obs_dim
        self.proprio_encoding_dim = proprio_encoding_dim
        self.vision_encoding_dim = vision_encoding_dim
        self.hidden_dim = hidden_dim

        self.proprio_encoder = ProprioNetwork(self.proprio_obs_dim, self.hidden_dim, self.proprio_encoding_dim).to(self.device)
        self.vision_encoder = AlexNetEncoder(self.vision_encoding_dim).to(self.device)
        self.critic = CoreNetwork(self.proprio_encoding_dim + self.vision_encoding_dim, self.hidden_dim, 1).to(self.device)
        self.actor_mean = PolicyNetwork(self.proprio_encoding_dim + self.vision_encoding_dim, self.hidden_dim,
                                        self.action_space).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_space)).to(self.device)

        self._proprio_keys = [
            'walker/joints_pos', 'walker/joints_vel', 'walker/tendons_pos',
            'walker/tendons_vel', 'walker/appendages_pos', 'walker/world_zaxis',
            'walker/sensors_accelerometer', 'walker/sensors_velocimeter',
            'walker/sensors_gyro', 'walker/sensors_touch',
        ]
        self._pixel_keys = ('walker/egocentric_camera',)

    def filter_obs(self, obs):
        proprio_obs = np.concatenate([obs[key] for key in self._proprio_keys], axis=-1)
        vision_obs = np.concatenate([obs[key] for key in self._pixel_keys], axis=-1)
        return proprio_obs, vision_obs

    def get_encoding(self, proprio_obs, vision_obs):
        # proprio_obs, vision_obs = self.filter_obs(obs)
        proprio_encoding = self.proprio_encoder(proprio_obs)
        # reshape vision to (batch_size, 3, 64, 64)
        vision_obs = vision_obs.reshape(-1, 3, 64, 64)
        vision_encoding = self.vision_encoder(vision_obs)
        return proprio_encoding, vision_encoding

    def get_value(self, proprio_obs, vision_obs):
        proprio_encoding, vision_encoding = self.get_encoding(proprio_obs, vision_obs)
        x = torch.cat([proprio_encoding, vision_encoding], dim=-1)
        value = self.critic(x)
        return value

    def get_action_and_value(self, proprio_obs, vision_obs, action=None):
        proprio_encoding, vision_encoding = self.get_encoding(proprio_obs, vision_obs)
        x = torch.cat([proprio_encoding, vision_encoding], dim=-1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        critic_value = self.critic(x)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), critic_value

    def save_checkpoint(self, env_name, suffix='', base_path='.'):
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), f'{base_path}/{env_name}_ppo_{suffix}.pth')

    def load_checkpoint(self, checkpoint_path, evaluate=False):
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(checkpoint_path))
        if evaluate:
            self.eval()
        else:
            self.train()
            # make sure the encoder is frozen
            for param in self.proprio_encoder.encoder.parameters():
                param.requires_grad = False

# ==================================================
from dm_control_suite_helpers import Rodent, _build_rodent_escape_env, _build_rodent_corridor_gaps, \
    _build_rodent_two_touch_env, _build_rodent_maze_env
# try out the wrappers
from dm_control_suite_helpers import FilterObservationsWrapper, NormilizeActionSpecWrapper, MujocoActionNormalizer
from acme import wrappers
import tree


def setup_rodent_env(task_name='escape', vision=False):
    if task_name == 'escape':
        env = _build_rodent_escape_env()
    elif task_name == 'gaps':
        env = _build_rodent_corridor_gaps()
    elif task_name == 'two_touch':
        env = _build_rodent_two_touch_env()
    elif task_name == 'mazes':
        env = _build_rodent_maze_env()
    elif task_name == 'test_cartpole':
        from dm_control import suite
        env = suite.load(domain_name="cartpole", task_name="balance")
    elif task_name == 'test_cheetah':
        from dm_control import suite
        env = suite.load(domain_name="cheetah", task_name="run")
    elif task_name == 'gaps_vnl':
        from dm_control import suite
        import virtualneurolab as vnl
        env = suite.load(domain_name="rodent", task_name="hop_gaps")
    else:
        raise ValueError(f'Unknown task name: {task_name}')

    _proprio_keys = [
        'walker/joints_pos', 'walker/joints_vel', 'walker/tendons_pos',
        'walker/tendons_vel', 'walker/appendages_pos', 'walker/world_zaxis',
        'walker/sensors_accelerometer', 'walker/sensors_velocimeter',
        'walker/sensors_gyro', 'walker/sensors_touch',
    ]
    _pixel_keys = ('walker/egocentric_camera',)

    # wrap the environment
    env = NormilizeActionSpecWrapper(env)
    env = MujocoActionNormalizer(environment=env, rescale='clip')
    env = wrappers.SinglePrecisionWrapper(env)

    # add filter wrapper if needed
    if task_name not in ['test_cartpole', 'test_cheetah']:
        all_observations = list(_proprio_keys)
        if vision:
            all_observations += list(_pixel_keys)

        env = FilterObservationsWrapper(env, all_observations)

    return env

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print('device:', device)
    print('cuda available:', torch.cuda.is_available())
    print('cuda args:', args.cuda)

    # TODO: later: parallelize envs
    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # import the


    environment_builder = lambda: setup_rodent_env(task_name=args.env_id, vision=True)

    from distributed import Sequential

    envs = Sequential(
        environment_builder, max_episode_steps=1000,
        workers=args.num_envs,
        name=args.env_id, vision=True, vision_key='walker/egocentric_camera')


    # get the action space
    action_space = np.prod(envs.single_action_space.shape)
    # get the observation space


    # agent = VisionAgent(envs).to(device)
    ### AHHHH HELP
    agent = VisionAgent(action_space, proprio_obs_dim=107, vision_obs_dim=(64, 64, 3), proprio_encoding_dim=16, vision_encoding_dim=16,
                 hidden_dim=128, use_vision=True, device=device).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume_checkpoint:  # todo: optimizer
        agent.load_checkpoint(args.resume_checkpoint)
        print('Resuming from checkpoint {}'.format(args.resume_checkpoint))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    ####
    vision_obs = torch.zeros((args.num_steps, args.num_envs) + (64, 64, 3)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs = torch.Tensor(envs.reset()).to(device)
    # next_proprio_obs, next_vision_obs = torch.Tensor(envs.reset()).to(device)
    next_proprio_obs, next_vision_obs = envs.reset()
    #### convert to tensors before feeding
    next_proprio_obs = torch.Tensor(next_proprio_obs).to(device)
    # stack the vision observations in numpy first
    next_vision_obs = np.vstack(next_vision_obs)
    next_vision_obs = torch.Tensor(next_vision_obs).to(device)


    # ERROR: ValueError: expected sequence of length 107 at dim 2 (got 64)

    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            obs[step] = next_proprio_obs
            vision_obs[step] = next_vision_obs

            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_proprio_obs, next_vision_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TODO: deal with parallel envs
            # TRY NOT TO MODIFY: execute the game and log data.
            observations, infos = envs.step(action.cpu().numpy())
            ####
            observations, vision_observations = observations

            #### todo: next_obs is kinda useless
            next_obs, reward, done = infos['observations'], infos['rewards'], infos['terminations']
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            ####
            next_proprio_obs = infos['observations']
            next_vision_obs = infos['vision_observations']
            #### convert to tensors before feeding
            next_proprio_obs = torch.Tensor(next_proprio_obs).to(device)

            # stack the vision observations in numpy first
            next_vision_obs = np.vstack(next_vision_obs)
            next_vision_obs = torch.Tensor(next_vision_obs).to(device)

            # whenever we finish, we need to make sure the episode info is recorded
            for i, item in enumerate(
                    infos['resets']):  # there are situations where there is a reset but no termination...
                if item:
                    episodic_return = infos['episodic_return'][i]
                    episodic_length = infos['episodic_length'][i]
                    print(
                        f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step=global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step=global_step)

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_proprio_obs, next_vision_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_vis_obs = vision_obs.reshape((-1,) + (64, 64, 3))

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_vis_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if update % 1 == 0:
            # torch.save(agent.state_dict(), "ppo_agent.pt")
            # save the policy with save_checkpoint
            checkpoint_name = str(args.seed) + "_" + str(num_updates)
            if args.resume_checkpoint:
                checkpoint_name += "_resumed"
            agent.save_checkpoint(args.env_id, checkpoint_name, base_path=args.save_directory)

    envs.close()
    writer.close()
