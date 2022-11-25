import argparse
import random

import torchvision as tv
import yaml
from PIL import Image
from gym3 import ViewerWrapper, VideoRecorderWrapper, ToBaselinesVecEnv
from procgen import ProcgenGym3Env

from agents.ppo import PPO
from common import set_global_seeds, set_global_log_levels
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common.storage import Storage


def to_int8(obs_arr):
    """Converts observations to int8 of range [0, 255], if they aren't already."""
    if obs_arr.dtype == float:
        if obs_arr.min() < 0 or obs_arr.max() > 1:
            raise ValueError("Float observations are not in [0, 1]!")
        obs_arr = (obs_arr * 255).astype(np.uint8)
    return obs_arr


def _save_trajectories(args, obs_traj_list, acts_list, infos_list, dones_list,
                       rew_list, final_obs):
    """Saves trajectories to disk. Add final_obs, which is the very last observation,
    i.e. the very last next_obs, that the agent has not seen yet, however it is
    necessary for the reward nets. We can't directly add this to the obs_traj_list,
    because then it would be duplicated, if this is not actually the last time we save
    trajectories, since at this same observation will be added to the list at the
    start of the next iteration."""
    if args.traj_path is not None:
        # Save trajectories to disk
        acts_concat = np.concatenate(acts_list)
        # Actions are saved as floats but are actually integer valued.
        acts_concat_int = acts_concat.astype(np.int8)
        if (acts_concat != acts_concat_int).any():
            raise ValueError("Actions are not integer valued!")

        # Turn dones into int
        dones_concat = np.concatenate(dones_list).astype(np.int8)

        # Calculate indices as expected by imitation.
        # The length of a trajectory is the number of transitions, which is the number
        # of observations minus 1, because the last observation is always the final obs
        # of an episode, which is not a transition.
        # The last trajectory may or may not be terminal, either way we don't need it
        # for the last index. (The last index splits the second to last from the last
        # trajectory, so the length of the last trajectory is irrelevant.)
        indices = np.cumsum([len(traj) - 1 for traj in obs_traj_list[:-1]])

        # Observations are simply all the observations concatenated.
        # We also add the final next_obs.
        obs_concat = np.concatenate(obs_traj_list + [[final_obs]])

        num_terminal_trajs = np.sum(dones_concat)
        # For every done we have one terminal trajectory.
        terminal = [True] * num_terminal_trajs
        if dones_concat[-1]:
            # The last trajectory is terminal if the last done is True.
            # That means we have only terminal trajectories, so we don't need to change
            # the terminal list.
            pass
        else:
            # The last trajectory is not terminal.
            # We need to add a False to the end of the terminal list.
            terminal.append(False)

        # Sanity check that the number of trajectories agrees.
        assert len(terminal) == len(obs_traj_list)

        # Sanity check that the number of transitions is correct.
        # For every trajectory, there is one more observation than transition, because
        # there is always a next_obs, even if the trajectory is not terminal.
        # The number of transitions is simply the number of actions.
        assert len(obs_concat) == len(acts_concat) + len(obs_traj_list)

        # Interesting thing to note is that imitation will expect indices to not line
        # up with the actual indices of the observations because it will then
        # automatically compensate for this fact.
        # This is due to the last next_obs being part of the obs list, hoever indices
        # are based on the length of the trajectories, which doesn't include the last
        # obs.
        condensed = {
            "obs": obs_concat,
            "acts": acts_concat_int,
            "infos": np.concatenate(infos_list),
            "terminal": np.array(terminal),
            "rews": np.concatenate(rew_list),
            "indices": np.array(indices),
        }

        total_timesteps = len(acts_concat_int)
        # This just adjusts the file name to include the number of timesteps (in 1000s).
        save_path = args.traj_path
        split = save_path.split(".")
        split[-2] += f"_{total_timesteps / 1000:.1f}k"
        save_path = ".".join(split)
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **condensed)

        os.replace(tmp_path, save_path)
        print(
            f"Saved trajectories to {save_path} for {total_timesteps} timesteps."
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='render',
                        help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='environment ID')
    parser.add_argument('--start_level', type=int, default=int(0),
                        help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0),
                        help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='hard',
                        help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200',
                        help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cpu', required=False,
                        help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False,
                        help='visible device in CUDA')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999),
                        help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1),
                        help='number of checkpoints to store')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=int(1000),
                        help='number of episodes to render / save')

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

    # render parameters
    parser.add_argument('--tps', type=int, default=15, help="env fps")
    parser.add_argument('--vid_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_value', action='store_true')
    parser.add_argument('--save_value_individual', action='store_true')
    parser.add_argument('--value_saliency', action='store_true')

    # Saving trajectories (compatible with HumanCompatibleAI/imitation)
    parser.add_argument('--traj_path', type=str, default=None)

    parser.add_argument('--random_percent', type=float, default=0.,
                        help='percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--corruption_type', type=str, default=None)
    parser.add_argument('--corruption_severity', type=str, default=1)
    parser.add_argument('--agent_view', action="store_true",
                        help="see what the agent sees")
    parser.add_argument('--continue_after_coin', action="store_true",
                        help="level doesnt end when agent gets coin")
    parser.add_argument('--noview', action="store_true", help="just take vids")

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    if args.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_envs = 1


    def create_venv_render(args, hyperparameters, is_valid=False):
        venv = ProcgenGym3Env(num=n_envs,
                              env_name=args.env_name,
                              num_levels=0 if is_valid else args.num_levels,
                              start_level=0 if is_valid else args.start_level,
                              distribution_mode=args.distribution_mode,
                              num_threads=1,
                              render_mode="rgb_array",
                              random_percent=args.random_percent,
                              corruption_type=args.corruption_type,
                              corruption_severity=int(args.corruption_severity),
                              continue_after_coin=args.continue_after_coin,
                              )
        info_key = None if args.agent_view else "rgb"
        ob_key = "rgb" if args.agent_view else None
        if not args.noview:
            venv = ViewerWrapper(venv, tps=args.tps, info_key=info_key,
                                 ob_key=ob_key)  # N.B. this line caused issues for me. I just commented it out, but it's uncommented in the pushed version in case it's just me (Lee).
        if args.vid_dir is not None:
            venv = VideoRecorderWrapper(venv, directory=args.vid_dir,
                                        info_key=info_key, ob_key=ob_key, fps=args.tps)
        venv = ToBaselinesVecEnv(venv)
        venv = VecExtractDictObs(venv, "rgb")
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
            # the img frames
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)

        return venv


    n_steps = hyperparameters.get('n_steps', 256)

    # env = create_venv(args, hyperparameters)
    # env_valid = create_venv(args, hyperparameters, is_valid=True)
    env = create_venv_render(args, hyperparameters, is_valid=True)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    if args.logdir is None:
        logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'RENDER_seed' + '_' + \
                 str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('logs', logdir)
    else:
        logdir = args.logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir_indiv_value = os.path.join(logdir, 'value_individual')
    if not (os.path.exists(logdir_indiv_value)) and args.save_value_individual:
        os.makedirs(logdir_indiv_value)
    logdir_saliency_value = os.path.join(logdir, 'value_saliency')
    if not (os.path.exists(logdir_saliency_value)) and args.value_saliency:
        os.makedirs(logdir_saliency_value)
    print(f'Logging to {logdir}')
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    # storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        agent = PPO(env, policy, logger, storage, device, num_checkpoints,
                    **hyperparameters)
    else:
        raise NotImplementedError

    agent.policy.load_state_dict(
        torch.load(args.model_file, map_location=device)["model_state_dict"])
    agent.n_envs = n_envs


    ############
    ## RENDER ##
    ############

    # save observations and value estimates
    def save_value_estimates(storage, epoch_idx):
        """write observations and value estimates to npy / csv file"""
        print(f"Saving observations and values to {logdir}")
        np.save(logdir + f"/observations_{epoch_idx}", storage.obs_batch)
        np.save(logdir + f"/value_{epoch_idx}", storage.value_batch)
        return


    def save_value_estimates_individual(storage, epoch_idx, individual_value_idx):
        """write individual observations and value estimates to npy / csv file"""
        print(f"Saving random samples of observations and values to {logdir}")
        obs = storage.obs_batch.clone().detach().squeeze().permute(0, 2, 3, 1)
        obs = (obs * 255).cpu().numpy().astype(np.uint8)
        vals = storage.value_batch.squeeze()

        random_idxs = np.random.choice(obs.shape[0], 5, replace=False)
        for rand_id in random_idxs:
            im = obs[rand_id]
            val = vals[rand_id]
            im = Image.fromarray(im)
            im.save(logdir_indiv_value + f"/obs_{individual_value_idx:05d}.png")
            np.save(logdir_indiv_value + f"/val_{individual_value_idx:05d}.npy", val)
            individual_value_idx += 1
        return individual_value_idx


    def write_scalar(scalar, filename):
        """write scalar to filename"""
        with open(logdir + "/" + filename, "w") as f:
            f.write(str(scalar))


    obs = agent.env.reset()
    hidden_state = np.zeros((agent.n_envs, agent.storage.hidden_state_size))
    done = np.zeros(agent.n_envs)

    # The number of iterations depends on the number of episodes we want and the number
    # of parallel environments we are using
    num_iterations = args.num_episodes // agent.n_envs
    if args.num_episodes % agent.n_envs != 0:
        num_iterations += 1  # We want *at least* num_episodes

    print("Start running collection of episodes.")
    # For saving trajectories / transitions
    obs_traj_list = [[]]
    current_traj = 0
    acts_list = []
    infos_list = []
    rew_list = []
    dones_list = []

    save_rollouts_every = num_iterations // 10
    if num_iterations % 10 != 0:
        # If it doesn't add up exactly save rollouts less frequently
        save_rollouts_every += 1

    individual_value_idx = 1001
    save_frequency = 1
    saliency_save_idx = 0
    epoch_idx = 0
    total_steps = 0
    for iteration in range(num_iterations):
        print(f"Iterations {iteration + 1}/{num_iterations}")
        agent.policy.eval()
        for _ in range(agent.n_steps):  # = 256
            if args.traj_path is not None:
                # Before anything we save the obs (if we are saving trajectories).
                obs_copy = obs[0].copy()
                obs_copy = to_int8(obs_copy).transpose(1, 2, 0)
                obs_traj_list[current_traj].append(obs_copy)

            if not args.value_saliency:
                act, log_prob_act, value, next_hidden_state = agent.predict(obs,
                                                                            hidden_state,
                                                                            done)
            else:
                act, log_prob_act, value, next_hidden_state, value_saliency_obs = agent.predict_w_value_saliency(
                    obs, hidden_state, done)
                if saliency_save_idx % save_frequency == 0:
                    value_saliency_obs = value_saliency_obs.swapaxes(1, 3)
                    obs_copy = obs.swapaxes(1, 3)

                    ims_grad = value_saliency_obs.mean(axis=-1)

                    percentile = np.percentile(np.abs(ims_grad), 99.9999999)
                    ims_grad = ims_grad.clip(-percentile, percentile) / percentile
                    ims_grad = torch.tensor(ims_grad)
                    blurrer = tv.transforms.GaussianBlur(
                        kernel_size=5,
                        sigma=5.)  # (5, sigma=(5., 6.))
                    ims_grad = blurrer(ims_grad).squeeze().unsqueeze(-1)

                    pos_grads = ims_grad.where(ims_grad > 0.,
                                               torch.zeros_like(ims_grad))
                    neg_grads = ims_grad.where(ims_grad < 0.,
                                               torch.zeros_like(ims_grad)).abs()

                    # Make a couple of copies of the original im for later
                    sample_ims_faint = torch.tensor(obs_copy.mean(-1)) * 0.2
                    sample_ims_faint = torch.stack([sample_ims_faint] * 3, axis=-1)
                    sample_ims_faint = sample_ims_faint * 255
                    sample_ims_faint = sample_ims_faint.clone().detach().type(
                        torch.uint8).cpu().numpy()

                    grad_scale = 9.
                    grad_vid = np.zeros_like(sample_ims_faint)
                    pos_grads = pos_grads * grad_scale * 255
                    neg_grads = neg_grads * grad_scale * 255
                    grad_vid[:, :, :, 2] = pos_grads.squeeze().clone().detach().type(
                        torch.uint8).cpu().numpy()
                    grad_vid[:, :, :, 0] = neg_grads.squeeze().clone().detach().type(
                        torch.uint8).cpu().numpy()

                    grad_vid = grad_vid + sample_ims_faint

                    grad_vid = Image.fromarray(grad_vid.swapaxes(0, 2).squeeze())
                    grad_vid.save(
                        logdir_saliency_value + f"/sal_obs_{saliency_save_idx:05d}_grad.png")

                    obs_copy = (obs_copy * 255).astype(np.uint8)
                    obs_copy = Image.fromarray(obs_copy.swapaxes(0, 2).squeeze())
                    obs_copy.save(
                        logdir_saliency_value + f"/sal_obs_{saliency_save_idx:05d}_raw.png")

                saliency_save_idx += 1

            next_obs, rew, done, info = agent.env.step(act)
            total_steps += 1

            if done:
                # In gym3 when the end of the episode is reached the env is immediately
                # reset. That is why next_obs is already the first observation of the
                # next episode.
                # However, in our modifications to procgen the finals observation of the
                # previous episode is saved in the info dict. We access this here and
                # append it to the list of observation trajectories
                final_obs = info[0]["final_obs"].copy()
                # Unlike normal obs, these obs are already in teh format where the last
                # dim is for channels, so we don't need to transpose here.
                final_obs = to_int8(final_obs)
                obs_traj_list[current_traj].append(final_obs)
                # Now the trajectory is done, so we start a new one.
                current_traj += 1
                obs_traj_list.append([])
                # The first observation will be added to this new trajectory at the
                # beginning of the next iteration of this for loop.

            agent.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act,
                                value)
            obs = next_obs
            hidden_state = next_hidden_state

        _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
        agent.storage.store_last(obs, hidden_state, last_val)

        if args.save_value_individual:
            individual_value_idx = save_value_estimates_individual(agent.storage,
                                                                   epoch_idx,
                                                                   individual_value_idx)

        if args.save_value:
            save_value_estimates(agent.storage, epoch_idx)

        if args.traj_path is not None:
            # Flatten batches as they contain infos for each env
            for env in range(agent.storage.num_envs):

                # We copy because original tensors will be reused by the agent.
                acts_list.append(agent.storage.act_batch.numpy()[:, env].copy())
                # I don't bother to save the actual infos because I don't need them, and
                # they are formatted weirdly.
                infos_list.append(np.full(len(agent.storage.act_batch), {}).copy())
                rew_list.append(agent.storage.rew_batch.numpy()[:, env].copy())
                dones_list.append(agent.storage.done_batch.numpy()[:, env].copy())

        epoch_idx += 1

        agent.storage.compute_estimates(agent.gamma, agent.lmbda, agent.use_gae,
                                        agent.normalize_adv)

        if iteration != 0 and iteration % save_rollouts_every == 0:
            # The last next_obs has not been added to the obs_traj_list yet, so we
            # add it here. We can't add it to the list directly, because then it would
            # be added to the list twice.
            # At this point in the code the final next_obs is also obs, so we just use
            # that because then Python doesn't complain that it might not be defined
            # yet.
            final_obs_of_iteration = obs[0].copy()
            final_obs_of_iteration = to_int8(final_obs_of_iteration).transpose(1, 2, 0)
            _save_trajectories(args, obs_traj_list, acts_list, infos_list, dones_list,
                               rew_list, final_obs_of_iteration)
    # See above for explanation.
    final_obs_of_iteration = obs[0].copy()
    final_obs_of_iteration = to_int8(final_obs_of_iteration).transpose(1, 2, 0)
    _save_trajectories(args, obs_traj_list, acts_list, infos_list, dones_list, rew_list, final_obs_of_iteration)
