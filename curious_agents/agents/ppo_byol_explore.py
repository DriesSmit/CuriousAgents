# Adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_rnn.py
# Please visit the repo above and support the authors.
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
import jumanji
from jumanji.wrappers import AutoResetWrapper
import chex


def process_observation(observation, time_limit):
    """
    Add the agent and the target to the walls array.
    Turn the observation into an 3D array
    Adapted from jumanji's process_observation function

    Args:
        observation: raw observation
        time_limit: TODO

    Returns:
        Processed observation
    """
    agent = 2
    target = 3
    obs = observation.walls.astype(int)
    obs = obs.at[tuple(observation.agent_position)].set(agent)
    obs = obs.at[tuple(observation.target_position)].set(target)

    # Determine the number of unique classes
    n_classes = target + 1  # assuming classes start at 0

    # One-hot encode the observations
    one_hot_obs = jax.nn.one_hot(obs, n_classes)

    # Add step count layer
    step_count = np.ones(obs.shape) * observation.step_count / time_limit

    # Concatenate the one-hot encoded observations with the step count
    obs = jnp.concatenate([one_hot_obs, step_count[..., None]], axis=-1)

    return obs

class ObservationEncoder(nn.Module):
    latent_size: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        """
        Converts raw observations into latent representations.
        Neural network consisting of 3 convolutional layers and 2 dense layers. 

        Args:
            x: Raw observation

        Returns:
            Latent representation of raw observation
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Convolutional layers
        layer_out = x
        for _ in range(3):
            layer_out = nn.Conv(
                features=32,  # increased the number of features
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",  # added padding
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(layer_out)
            layer_out = activation(layer_out)

        layer_out = layer_out.reshape((layer_out.shape[0], -1))

        # Dense layers
        for _ in range(2):
            layer_out = nn.Dense(
                128,  # increased the number of features
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(layer_out)
            layer_out = activation(layer_out)

        # Reshape output to fit latent dimension size
        layer_out = nn.Dense(
            self.latent_size,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(layer_out)

        # tanh activation the output for stability
        layer_out = nn.tanh(layer_out)

        return layer_out

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    latent_size: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        """
        Generate policy over actions and compute value of current state

        Args:
            x: Current state

        Returns:
            Policy for state (from actor) and state value (from critic)
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor - produce policy over actions using ObservationEncoder
        policy_obs_encoder = ObservationEncoder(self.latent_size, self.activation)
        actor_mean = policy_obs_encoder(x)

        # Actor - dense layers
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Actor - categorical distribution represents policy
        pi = distrax.Categorical(logits=actor_mean)

        # Critic - computes value of state using separate ObservationEncoder
        critic_obs_encoder = ObservationEncoder(self.latent_size, self.activation)
        critic = critic_obs_encoder(x)

        # Critic - dense layers
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # Return policy and state value
        return pi, jnp.squeeze(critic, axis=-1)

class WorldModel(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, latent_in, action):
        """
        Models transitions in the environment

        Arguments:
            latent_in: latent representation of an observation
            action: agent action as a categorical distribution

        Returns:
            TODO
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # One-hot encode the action and concatenate with latent representation
        one_hot_action = jax.nn.one_hot(action, self.action_dim)
        inp = jnp.concatenate([latent_in, one_hot_action], axis=-1)

        # 3 Dense layers
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            latent_in.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(layer_out)
        return layer_out

class Transition(NamedTuple):
    """Contains information for an environment transition"""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

class BYOLTrainState(NamedTuple):
    """Contains BYOL-Explore training parameters"""
    policy: jnp.array
    online: jnp.array
    target: jnp.array
    world_model: jnp.array

def flatten_params(params):
    """Flatten a dictionary of parameters into a vector."""
    # Concatenate all parameter arrays into a single vector
    return jnp.concatenate(
        [jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)]
    )

def l2_norm_squared(arr, axis=-1):
    """Compute squared L2 norm of array"""
    # TODO: Should this be a mean instead for scalability?
    return jnp.sum(jnp.square(arr), axis=axis)

def compute_distance(arr1, arr2, axis=-1):
    """Compute the Euclidean distance between two sets of arrays."""
    return jnp.linalg.norm(arr1 - arr2, axis=axis)

def normalise(arr):
    """Normalise an array using the L2 norm."""
    norm = jnp.sqrt(jnp.sum(jnp.square(arr), axis=-1))[..., None]
    return arr / norm

def byol_loss(pred_l_t, l_t):
    """
    Compute BYOL loss by measuring difference between predicted and actual
    latent states

    Args:
        pred_l_t: Normalised predicted latent state
        l_t: Normalised actual latent state

    Returns:
        Distance (Euclidean) between predicted and actual latent states
    """
    # CALCULATE WORLD MODEL LOSS
    # TODO: ARE LATENT STATES ALREADY NORMALISED?
    norm_pred_l_t = pred_l_t  # normalise(pred_l_t)
    norm_l_t = l_t  # normalise(l_t)

    # Cap the world model loss
    return compute_distance(norm_pred_l_t, norm_l_t)
    # return l2_norm_squared(norm_pred_l_t-norm_l_t)


class PPOAgent:
    def __init__(self, env_name) -> None:
        self._config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 128,
            "UPDATE_EPOCHS": 4,
            "NUM_MINIBATCHES": 4,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.15,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "TARGET_UPDATE_RATE": 0.001,
            "LATENT_SIZE": 64,
            "REWARD_UPDATE_RATE": 0.001,
            "ACTIVATION": "relu",
            "ENV_NAME": env_name,
            "ANNEAL_LR": False,
            "DEBUG": True,
        }
        self._config["MINIBATCH_SIZE"] = (
            self._config["NUM_ENVS"]
            * self._config["NUM_STEPS"]
            // self._config["NUM_MINIBATCHES"]
        )

        # Hardcoded 2D environment size - 5x5 grid
        generator = jumanji.environments.routing.maze.generator.RandomGenerator(
            num_rows=5, num_cols=5
        )
        self._env = jumanji.make(self._config["ENV_NAME"], generator=generator)
        self._env = AutoResetWrapper(self._env)

        # INIT NETWORKS
        num_actions = self._env.action_spec().num_values

        # Policy Network - Actor Critic network producing policy and state value
        self._policy_network = ActorCritic(
            num_actions,
            self._config["LATENT_SIZE"],
            activation=self._config["ACTIVATION"],
        )

        # Online Encoder - Encode observations into latent representations
        self._online_encoder = ObservationEncoder(
            latent_size=self._config["LATENT_SIZE"],
            activation=self._config["ACTIVATION"],
        )

        # Target Encoder - Mirror of online encoder
        self._target_encoder = ObservationEncoder(
            latent_size=self._config["LATENT_SIZE"],
            activation=self._config["ACTIVATION"],
        )

        # World Model - Predict future latent representations
        self._world_model = WorldModel(
            num_actions, activation=self._config["ACTIVATION"]
        )

        # INIT LOGGER
        self._logger = None

    def init_state(self, rng):
        """Initialise agent state using PRNG key"""
        # def linear_schedule(count):
        #     frac = (
        #         1.0
        #         - (count // (self._config["NUM_MINIBATCHES"] * self._config["UPDATE_EPOCHS"]))
        #         / self._config["NUM_UPDATES"]
        #     )
        #     return self._config["LR"] * frac

        # Split provided PRNG key
        rng, _rng = jax.random.split(rng)

        # Reset environment to initial state specified by PRNG key
        reset_rng = jax.random.split(_rng, self._config["NUM_ENVS"])
        env_state, timestep = jax.vmap(self._env.reset)(reset_rng)
        obs = jax.vmap(process_observation, in_axes=(0, None))(
            timestep.observation, self._env.time_limit
        )

        # Generate random PRNG keys for policy network, online and target
        # encoders, and world model
        rng, pol_rng, online_rng, target_rng, wm_rng = jax.random.split(rng, 5)
        
        # INIT THE POLICY NETWORK
        policy_params = self._policy_network.init(pol_rng, obs)

        # INIT THE ENCODERS
        online_params = self._online_encoder.init(online_rng, obs)
        target_params = self._target_encoder.init(target_rng, obs)

        # INIT THE WORLD MODEL
        latent_size = self._online_encoder.latent_size
        zero_latent = jnp.zeros(latent_size, dtype=jnp.float32)
        zero_action = jnp.zeros((), dtype=jnp.int32)
        wm_params = self._world_model.init(wm_rng, zero_latent, zero_action)

        # Define optimizers for policy network and world model
        if self._config["ANNEAL_LR"]:
            pass
            # pol_tx = optax.chain(
            #     optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            #     optax.adam(learning_rate=linear_schedule),
            # )
            # wm_tx = optax.chain(
            #     optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            #     optax.adam(learning_rate=linear_schedule),
            # )
            
        else:
            # Use Adam and clip gradients to prevent large updates
            pol_tx = optax.chain(
                optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
                optax.adam(self._config["LR"]),
            )
            wm_tx = optax.chain(
                optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
                optax.adam(self._config["LR"]),
            )
            
        # Define training state for policy network
        policy_train_state = TrainState.create(
            apply_fn=self._policy_network.apply,
            params=policy_params,
            tx=pol_tx,
        )

        # Define training state for online encoder
        online_train_state = TrainState.create(
            apply_fn=self._online_encoder.apply,
            params=online_params,
            tx=wm_tx,
        )

        # Define training state for target encoder
        wm_train_state = TrainState.create(
            apply_fn=self._world_model.apply,
            params=wm_params,
            tx=wm_tx,
        )

        # Encapsulate all training state into a single BYOL training state
        train_states = BYOLTrainState(
            policy=policy_train_state,
            online=online_train_state,
            target=target_params,
            world_model=wm_train_state,
        )

        # Return initial training states, environment state, observations, 
        # reward standard deviation, rng state, step count.
        step = 0
        reward_std = 1.0
        return (train_states, env_state, obs, reward_std, rng, step)

    def _env_step(self, runner_state, unused):
        """Defines agent's interaction with environment for a single timestep"""
        train_states, env_state, last_obs, reward_std, rng, step = runner_state

        # Obtain policy and state value from policy (Actor Critic) network
        pi, value = self._policy_network.apply(train_states.policy.params, last_obs)
        
        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # PERFORM STEP IN ENV
        env_state, timestep = jax.vmap(self._env.step, in_axes=(0, 0))(
            env_state, action
        )

        # Obtain timestep reward and check if the current run is done
        done = timestep.last()
        original_reward = timestep.reward

        # Turn the observation into an 3D array
        obs = jax.vmap(process_observation, in_axes=(0, None))(
            timestep.observation, self._env.time_limit
        )

        # Calcuate the distance between the predicted and the actual observation
        l_tm1 = self._online_encoder.apply(train_states.online.params, last_obs)
        pred_l_t = self._world_model.apply(
            train_states.world_model.params, l_tm1, action
        )

        # Get the latent state from the target network
        l_t = self._target_encoder.apply(train_states.target, obs)

        # Set the reward to be the distance between the predicted and the actual observation
        reward = byol_loss(pred_l_t, l_t)
        alpha = self._config["REWARD_UPDATE_RATE"]
        
        # TODO: This is not really the std
        reward_std = reward_std * (1 - alpha) + alpha * jnp.std(reward)
        reward = reward / reward_std

        # Update transition information
        info = {"step_rewards": original_reward, "wm_rewards": reward}
        transition = Transition(
            done, action, value, reward, log_prob, last_obs, obs, info
        )

        # Step once for every environment.
        runner_state = (
            train_states,
            env_state,
            obs,
            reward_std,
            rng,
            step + self._config["NUM_ENVS"],
        )
        
        return runner_state, (transition, env_state)

    # TRAIN LOOP
    def _update_step(self, runner_state):
        # RUN ENV - Obtain batch of transitions
        runner_state, (traj_batch, _) = jax.lax.scan(
            self._env_step, runner_state, None, self._config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        # Obtain value for last observation
        train_states, env_state, last_obs, reward_std, rng, step = runner_state
        _, last_val = self._policy_network.apply(train_states.policy.params, last_obs)

        def _calculate_gae(traj_batch, last_val):
            """
            Generalized Advantage Estimation

            Args:
                traj_batch: Batch of sampled trajectories
                last_val: Value for last observation obtained earlier
            """
            
            def _get_advantages(gae_and_next_value, transition):
                """Advantage Calculation"""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self._config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + self._config["GAMMA"]
                    * self._config["GAE_LAMBDA"]
                    * (1 - done)
                    * gae
                )
                return (gae, value), gae

            # Calculate advantages and combine with value predictions
            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # Combination with value predictions provide target values for value fn
        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            """
            Update actor models with minibatch updates

            Args:
                update_state: Contains information for model updates
            """
            def _update_minbatch(train_states, batch_info):
                """
                Use minibatch to perform updates to agent models

                Args:
                    train_states: TODO
                    batch_info: TODO

                Returns:
                    Training states, policy loss, world model loss, and
                    distance metric between online and target model params
                """
                traj_batch, advantages, targets = batch_info

                def _agent_loss_fn(params, traj_batch, gae, targets):
                    """
                    Calculate Agent Loss

                    Args:
                        params: Policy network parameters
                        traj_batch: Sampled trajectories batch
                        gae: Advantages from _calculate_gae()
                        targets: Targets from _calculate_gae()

                    Returns:
                        Total loss - weighted sum of actor, value and entropy losses
                    """
                    # RERUN NETWORKS
                    pi, value = self._policy_network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS - Ensure value prediction is clipped 
                    # to prevent large updates
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-self._config["CLIP_EPS"], self._config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS - Ensure update is clipped
                    # Ratio to calculate current vs old policy deviation
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - self._config["CLIP_EPS"],
                            1.0 + self._config["CLIP_EPS"],
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    
                    # Calculate entropy loss - penalize deterministic policies
                    entropy_loss = -pi.entropy().mean()

                    # Calculate total loss - weighted sum of actor, value and
                    # entropy losses
                    total_loss = (
                        actor_loss
                        + self._config["VF_COEF"] * value_loss
                        + self._config["ENT_COEF"] * entropy_loss
                    )
                    return total_loss, (value_loss, actor_loss, entropy_loss)

                # UPDATE THE POLICY - Compute gradients w.r.t policy parameters
                grad_fn = jax.value_and_grad(_agent_loss_fn, has_aux=True)
                pol_loss, grads = grad_fn(
                    train_states.policy.params, traj_batch, advantages, targets
                )
                new_policy_state = train_states.policy.apply_gradients(grads=grads)

                def _wm_loss_fn(online_params, world_model_params, traj_batch):
                    """
                    Compute BYOL loss between 
                        - predicted latent representation states (world model) 
                        - actual latent representation states (target encoder)
                    """
                    # RERUN NETWORKS - Get predicted latent states
                    l_tm1 = self._online_encoder.apply(online_params, traj_batch.obs)
                    pred_l_t = self._world_model.apply(
                        world_model_params, l_tm1, traj_batch.action
                    )
                    
                    # Get actual latent states
                    l_t = jax.lax.stop_gradient(
                        self._target_encoder.apply(
                            train_states.target, traj_batch.next_obs
                        )
                    )
                    
                    # Return loss between predicted and actual latent states
                    return byol_loss(pred_l_t, l_t).mean()

                # UPDATE THE WORLD MODEL - Compute gradients w.r.t online
                # encoder and world model parameters
                grad_fn = jax.value_and_grad(_wm_loss_fn, argnums=[0, 1])
                (wm_loss, (online_grads, wm_grads),) = grad_fn(
                    train_states.online.params,
                    train_states.world_model.params,
                    traj_batch,
                )

                # Update both online encoder and world model
                new_online_state = train_states.online.apply_gradients(
                    grads=online_grads
                )
                new_wm_state = train_states.world_model.apply_gradients(grads=wm_grads)

                # UPDATE THE TARGET MODEL USING MOVING AVERAGES (EMA)
                alpha = self._config["TARGET_UPDATE_RATE"]
                new_target_state = jax.tree_util.tree_map(
                    lambda target, online: (1 - alpha) * target + alpha * online,
                    train_states.target,
                    train_states.online.params,
                )
                
                # Calculate the distance metrix between the online and target model
                # STEP 1: Flatten both models
                online_params_flat = flatten_params(train_states.online.params)
                target_params_flat = flatten_params(train_states.target)

                # STEP 2: Calculate the distance
                distance = compute_distance(
                    online_params_flat,
                    target_params_flat,
                )

                train_states = BYOLTrainState(
                    policy=new_policy_state,
                    online=new_online_state,
                    world_model=new_wm_state,
                    target=new_target_state,
                )

                return train_states, [pol_loss, wm_loss, distance]

            # Extract update information from update state
            train_states, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = (
                self._config["MINIBATCH_SIZE"] * self._config["NUM_MINIBATCHES"]
            )
            assert (
                batch_size == self._config["NUM_STEPS"] * self._config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            
            # Reshape and divide trajectories into minibtaches
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [self._config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            
            # Apply update minibatch function to every minibatch
            train_states, metrics = jax.lax.scan(
                _update_minbatch, train_states, minibatches
            )
            return (train_states, traj_batch, advantages, targets, rng), metrics

        # Create update state and perform actor models update
        update_state = (train_states, traj_batch, advantages, targets, rng)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, self._config["UPDATE_EPOCHS"]
        )

        # Extract metrics after update conclusion
        (
            (total_loss, (value_loss, actor_loss, entropy_loss)),
            wm_loss,
            target_dist,
        ) = metrics
        reward_std = runner_state[-3]
        metric_info = {
            "total_loss": total_loss.mean(),
            "value_loss": value_loss.mean(),
            "actor_loss": actor_loss.mean(),
            "entropy_loss": entropy_loss.mean(),
            "wm_loss": wm_loss.mean(),
            "target_dist": target_dist.mean(),
            "reward_std": reward_std,
        }

        # Obtain current training state and extrinsic, intrinsic rewards
        train_states = update_state[0]
        step_rewards = traj_batch.info["step_rewards"]
        wm_rewards = traj_batch.info["wm_rewards"]

        # Calculate mean of extrinsic and intrinsic episode rewards
        episode_rewards = jnp.sum(step_rewards) / jnp.sum(traj_batch.done)
        episode_wm_rewards = jnp.sum(wm_rewards) / jnp.sum(traj_batch.done)

        # Debugging functionality - log metrics for each step
        rng = update_state[-1]
        if self._config.get("DEBUG"):

            def callback(episode_rewards, episode_wm_rewards, metric_info, step):
                print(
                    "Timestep: {}. Episode return: {:.2f}.".format(
                        step, episode_rewards
                    )
                )

                self._logger.write("episode_rewards", episode_rewards, step=step)
                self._logger.write("episode_wm_rewards", episode_wm_rewards, step=step)
                self._logger.write("total_loss", metric_info["total_loss"], step=step)
                self._logger.write("value_loss", metric_info["value_loss"], step=step)
                self._logger.write("actor_loss", metric_info["actor_loss"], step=step)
                self._logger.write(
                    "entropy_loss", metric_info["entropy_loss"], step=step
                )
                self._logger.write("wm_loss", metric_info["wm_loss"], step=step)
                self._logger.write(
                    "target_distance", metric_info["target_dist"], step=step
                )
                self._logger.write("reward_std", metric_info["reward_std"], step=step)

            jax.debug.callback(
                callback, episode_rewards, episode_wm_rewards, metric_info, step
            )

        # Update runner state with newest information and return
        runner_state = (train_states, env_state, last_obs, reward_std, rng, step)
        return runner_state

    def run_and_save_gif(self, runner_state, num_steps=1000, output_loc="./logs/Maze.gif"):
        """
        Execute agent training loop with visualisation

        Args:
            runner_state: State of agent at current timestep

        Keyword Arguments:
            num_steps: Number of training steps (default: {1000})
            output_loc: Destination of visualisation gif (default: {"./logs/Maze.gif"})
        """
        # RUN ENV
        print("Running env..")
        env_state_seq = []
        jitted_step_fn = jax.jit(self._env_step)
        for _ in range(num_steps):
            runner_state, _ = jitted_step_fn(runner_state, None)
            env_state_seq.append(runner_state[1])

        # Take first run for each array using JAX treemap
        env_state_seq = jax.tree_map(lambda x: x[0], env_state_seq)

        # VISUALIZE
        print("Visualizing env..")
        self._env.animate(env_state_seq, interval=150, save_path=output_loc)

    def run(self, runner_state, logger, external_rewards=True, steps=10000, evaluation=False):
        """
        Execute agent training loop

        Args:
            runner_state: State of agent at current timestep
            logger: Metrics logger
            external_rewards: Whether to use external rewards (default: {True})
            steps: Number of training steps (default: {10000})
            evaluation: Whether the following is an evaluation run (default: {False})

        Returns:
            The final state of the agent
        """
        # Set the logger
        self._logger = logger

        # TRAIN LOOP
        num_updates = steps // self._config["NUM_ENVS"] // self._config["NUM_STEPS"]

        def update_fn(runner_state, unused):
            return (self._update_step(runner_state), None)

        def scan_fn(runner_state):
            return jax.lax.scan(update_fn, runner_state, None, length=num_updates)

        # scan_fn = jax.jit(chex.assert_max_traces(scan_fn, n=1))
        # runner_state, unused = scan_fn(runner_state)
        runner_state, unused = jax.jit(scan_fn)(runner_state)

        return runner_state
