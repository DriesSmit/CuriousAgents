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
from jumanji.wrappers import AutoResetWrapper
from curious_agents.environments.minecraft2d.generator import RandomGenerator
from curious_agents.environments.minecraft2d.env import Minecraft2D
from curious_agents.environments.minecraft2d.constants import DIAMOND_ORE

# Turn the observation into an 3D array
# Adapted from jumanji's process_observation function
def process_observation(observation, time_limit):
    """Add the agent and the target to the walls array."""
    obs = observation.map.astype(int)
    n_classes = DIAMOND_ORE + 1  # assuming classes start at 0

    # One-hot encode the observations
    one_hot_obs = jax.nn.one_hot(obs, n_classes)

    # Add step count layer
    step_count = np.ones(obs.shape) * observation.step_count/time_limit

    # Concatenate the one-hot encoded observations with the step count
    obs = jnp.concatenate([one_hot_obs, step_count[..., None]], axis=-1)

    return obs

class ObservationEncoder(nn.Module):
    x_size: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Convolutional layers

        layer_out = x

        for _ in range(3):
            layer_out = nn.Conv(
                features=64,  # increased the number of features
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",  # added padding
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(layer_out)
            layer_out = activation(layer_out)

        layer_out = layer_out.reshape((layer_out.shape[0], -1))

        for _ in range(2):
            layer_out = nn.Dense(
                128,  # increased the number of features
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(layer_out)
            layer_out = activation(layer_out)

        layer_out = nn.Dense(
            self.x_size, 
            kernel_init=orthogonal(1.0), 
            bias_init=constant(0.0),
        )(layer_out)

        # tanh activation the output for stability
        # TODO: check if this is necessary
        layer_out = nn.tanh(layer_out)

        return layer_out
    
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    x_size: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, x_tm1):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_tm1)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic_obs_encoder = ObservationEncoder(self.x_size,  self.activation)
        critic = critic_obs_encoder(x)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
class WorldModel(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, z_t, x_tm1, a_tm1):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # One-hot encode the action
        one_hot_action = jax.nn.one_hot(a_tm1, self.action_dim)

        inp = jnp.concatenate([z_t, x_tm1, one_hot_action], axis=-1)

        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(x_tm1.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            layer_out
        )
        return layer_out

class Generator(nn.Module):
    z_dim: Sequence[int]
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x_tm1, a_tm1, x_t):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # One-hot encode the action
        one_hot_action = jax.nn.one_hot(a_tm1, self.action_dim)

        inp = jnp.concatenate([x_tm1, x_t, one_hot_action], axis=-1)

        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(self.z_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            layer_out
        )

        # tanh activation the output for stability
        # TODO: check if this is necessary
        layer_out = nn.tanh(layer_out)

        return layer_out

class Discriminator(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x_tm1, a_tm1, z_t):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # One-hot encode the action
        one_hot_action = jax.nn.one_hot(a_tm1, self.action_dim)

        inp = jnp.concatenate([z_t, x_tm1, one_hot_action], axis=-1)

        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            layer_out
        )

        # Limit the output to the range [-1, 1]
        # TODO: Does this even work? It seem hacky.
        # Remove this if possible.
        layer_out = jnp.tanh(layer_out)*5

        # Exponentiate the output to get a probability
        layer_out = jnp.exp(layer_out)

        return layer_out

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

class BOYLTrainState(NamedTuple):
        policy: jnp.array
        online: jnp.array
        target: jnp.array
        world_model: jnp.array
        generator: jnp.array
        discriminator: jnp.array

def flatten_params(params):
    """Flatten a dictionary of parameters into a vector."""
    # Concatenate all parameter arrays into a single vector
    return jnp.concatenate([jnp.reshape(p, (-1,)) for p in jax.tree_util.tree_leaves(params)])

def l2_norm_squared(arr, axis=-1):
    # TODO: Should this be a mean instead for scalability?
    return jnp.sum(jnp.square(arr), axis=axis)

def compute_distance(arr1, arr2,  axis=-1):
    """Compute the Euclidean distance between two sets of arrays."""
    return jnp.linalg.norm(arr1 - arr2, axis=axis)

def normalise(arr):
    """Normalise an array using the L2 norm."""
    norm = jnp.sqrt(jnp.sum(jnp.square(arr), axis=-1))[..., None]
    return arr/norm

def boyl_loss(pred_x_t, x_t):
    # CALCULATE WORLD MODEL LOSS
    norm_pred_x_t = pred_x_t # normalise(pred_x_t)
    norm_x_t = x_t # normalise(x_t)

    # Cap the world model loss
    # return compute_distance(norm_pred_x_t, norm_x_t)
    return l2_norm_squared(norm_pred_x_t-norm_x_t)

class PPOAgent():
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
        "DISC_IMP_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "TARGET_UPDATE_RATE": 0.001,
        "X_LATENT_SIZE": 64,
        "Z_LATENT_SIZE": 32,
        # "REWARD_UPDATE_RATE": 0.001,
        "ACTIVATION": "relu",
        "ENV_NAME": env_name,
        "DEBUG": True,
    }
        self._config["MINIBATCH_SIZE"] = (
            self._config["NUM_ENVS"] * self._config["NUM_STEPS"] // self._config["NUM_MINIBATCHES"]
        )

        generator = RandomGenerator(num_rows=5, num_cols=5)
        self._env = Minecraft2D(generator=generator)        
        self._env = AutoResetWrapper(self._env)

        # INIT NETWORKS
        num_actions = self._env.action_spec().num_values
        self._policy_network = ActorCritic(
            action_dim=num_actions, x_size=self._config["X_LATENT_SIZE"],
            activation=self._config["ACTIVATION"]
        )

        self._online_encoder = ObservationEncoder(
            x_size=self._config["X_LATENT_SIZE"], activation=self._config["ACTIVATION"]
        )

        self._world_model = WorldModel(
            action_dim=num_actions, activation=self._config["ACTIVATION"]
        )

        self._target_encoder = ObservationEncoder(
            x_size=self._config["X_LATENT_SIZE"], activation=self._config["ACTIVATION"]
        )

        self._generator = Generator(
            z_dim=self._config["Z_LATENT_SIZE"],
            action_dim=num_actions, activation=self._config["ACTIVATION"], 
        )

        self._discriminator = Discriminator(
            action_dim=num_actions,
            activation=self._config["ACTIVATION"]
        )

        

        # INIT LOGGER
        self._logger = None

    def init_state(self, rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self._config["NUM_ENVS"])

        env_state, timestep = jax.vmap(self._env.reset)(reset_rng)        
        obs = jax.vmap(process_observation, in_axes=(0, None))(timestep.observation, self._env.time_limit)

        # INIT RNG
        rng, online_rng, target_rng, pol_rng, wm_rng, gen_rng, disc_rng = jax.random.split(rng, 7)

        # INIT THE ENCODERS
        online_params = self._online_encoder.init(online_rng, obs)
        target_params = self._target_encoder.init(target_rng, obs)

        # Init latent
        x_zero = jnp.zeros(self._config["X_LATENT_SIZE"], dtype=jnp.float32)
        z_zero = jnp.zeros(self._config["Z_LATENT_SIZE"], dtype=jnp.float32)

        # INIT THE POLICY
        policy_params = self._policy_network.init(pol_rng, obs, x_zero)

        # INIT THE WORLD MODEL
        a_zero = jnp.zeros((), dtype=jnp.int32)
        wm_params = self._world_model.init(wm_rng, z_zero, x_zero, a_zero)

        # INIT THE GENERATOR
        gen_params = self._generator.init(gen_rng, x_zero, a_zero, x_zero)

        # INIT THE DISCRIMINATOR
        disc_params = self._discriminator.init(disc_rng, x_zero, a_zero, z_zero)
      
        pox_tx = optax.chain(
            optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            optax.adam(self._config["LR"]),
        )
        wm_tx = optax.chain(
            optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            optax.adam(self._config["LR"]),
        )
        policy_train_state = TrainState.create(
            apply_fn=self._policy_network.apply,
            params=policy_params,
            tx=pox_tx,
        )

        online_train_state = TrainState.create(
            apply_fn=self._online_encoder.apply,
            params=online_params,
            tx=wm_tx,
        )

        wm_train_state = TrainState.create(
            apply_fn=self._world_model.apply,
            params=wm_params,
            tx=wm_tx,
        )

        gen_train_state = TrainState.create(
            apply_fn=self._generator.apply,
            params=gen_params,
            tx=wm_tx,
        )

        dis_train_state = TrainState.create(
            apply_fn=self._discriminator.apply,
            params=disc_params,
            tx=wm_tx,
        )

        train_states = BOYLTrainState(
            policy=policy_train_state,
            online=online_train_state,
            target=target_params,
            world_model=wm_train_state,
            generator=gen_train_state,
            discriminator=dis_train_state,
        )

        step = 0
        reward_std = 1.0
        return (train_states, env_state, obs, reward_std, rng, step)
    
    def _env_step(self, runner_state, unused):
        train_states, env_state, last_obs, reward_std, rng, step = runner_state

        # Calculate the latent state
        x_tm1 = self._online_encoder.apply(train_states.online.params, last_obs)

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = self._policy_network.apply(train_states.policy.params, last_obs, x_tm1)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        env_state, timestep = jax.vmap(
            self._env.step, in_axes=(0, 0)
        )(env_state, action)

        # Turn the observation into an 3D array
        obs = jax.vmap(process_observation, in_axes=(0, None))(timestep.observation, self._env.time_limit)
        
        done = timestep.last()
        original_reward = timestep.reward

        # Calcuate the distance between the predicted and the actual observation
        x_t = self._target_encoder.apply(train_states.target, obs)
        z_t = self._generator.apply(train_states.generator.params, x_tm1, action, x_t) 
        pred_x_t = self._world_model.apply(train_states.world_model.params, z_t, x_tm1, action)

        # Set the reward to be the distance between the predicted and the actual observation
        reward = boyl_loss(pred_x_t, x_t)
        info = {"step_rewards": original_reward, "wm_rewards": reward}

        # alpha = self._config["REWARD_UPDATE_RATE"]
        # reward_std = reward_std*(1-alpha) + alpha*jnp.std(reward)
        # reward = reward/reward_std

        transition = Transition(
            done, action, value, reward, log_prob, last_obs, obs, info
        )
        
        # Step once for every environment.
        runner_state = (train_states, env_state, obs, reward_std, rng, step + self._config["NUM_ENVS"])
        return runner_state, (transition, env_state)

    # TRAIN LOOP
    def _update_step(self, runner_state):
        # RUN ENV
        runner_state, (traj_batch, _) = jax.lax.scan(
            self._env_step, runner_state, None, self._config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_states, env_state, last_obs, reward_std, rng, step = runner_state
        last_latent = self._online_encoder.apply(train_states.online.params, last_obs)
        _, last_val = self._policy_network.apply(train_states.policy.params, last_obs, last_latent)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self._config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + self._config["GAMMA"] * self._config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_states, batch_info):
                traj_batch, advantages, targets = batch_info

                def _policy_loss_fn(online_params, policy_params, traj_batch, gae, targets):
                    a_tm1 = traj_batch.action
                    o_tm1 = traj_batch.obs
                    v_tm1 = traj_batch.value

                    # RERUN NETWORKS
                    x_tm1 = self._online_encoder.apply(online_params, o_tm1)
                    pi, value = self._policy_network.apply(policy_params, o_tm1, x_tm1)
                    log_prob = pi.log_prob(a_tm1)
                    
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = v_tm1 + (
                        value - v_tm1
                    ).clip(-self._config["CLIP_EPS"], self._config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
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
                    entropy_loss = -pi.entropy().mean()

                    # CALCULATE TOTAL LOSS
                    total_loss = (
                        actor_loss
                        + self._config["VF_COEF"] * value_loss
                        + self._config["ENT_COEF"] * entropy_loss
                    )
                    return total_loss, (value_loss, actor_loss, entropy_loss)
                
                def _wm_loss_fn(online_params, world_model_params, traj_batch):
                    a_tm1 = traj_batch.action
                    o_tm1 = traj_batch.obs
                    o_t = traj_batch.next_obs

                    # RERUN NETWORKS
                    x_tm1 = self._online_encoder.apply(online_params, o_tm1)
                    x_t = jax.lax.stop_gradient(self._target_encoder.apply(train_states.target, o_t))
                    z_t = self._generator.apply(train_states.generator.params, x_tm1, a_tm1, x_t)
                    # TODO: Maybe add noise to z_t. This might encourage the world model to
                    # rely as much as possible on x_tm1 and a_tm1, and as little as possible
                    # on z_t.
                    pred_x_t = self._world_model.apply(world_model_params, z_t, x_tm1, a_tm1)
                    return boyl_loss(pred_x_t, x_t).mean()
                
                def calc_disc_loss(params, x_tm1, a_tm1, z_t):
                    # Entry fuction
                    entry_fn = jax.vmap(self._discriminator.apply, in_axes=(None, None, None, 0))

                    # Batch function
                    batch_fn = jax.vmap(entry_fn, in_axes=(None, 0, 0, None))

                    scores = batch_fn(params, x_tm1, a_tm1, z_t)
                    
                    # Get the diagonal of the matrix
                    sqeezed_scores = jnp.squeeze(scores)
                    diag_scores = jnp.diag(sqeezed_scores)
                    
                    ratios = diag_scores / (len(scores[0])*jnp.sum(sqeezed_scores, axis=-1))

                    log_ratios = jnp.log(ratios)

                    return -jnp.mean(log_ratios)

                def _generator_loss_fn(generator_params, traj_batch):
                    # TODO: There is a lot of code duplication here. Can we not 
                    # somehow combine this with the other loss functions?
                    a_tm1 = traj_batch.action
                    o_tm1 = traj_batch.obs
                    o_t = traj_batch.next_obs
                    x_tm1 = jax.lax.stop_gradient(self._online_encoder.apply(train_states.online.params, o_tm1))
                    x_t = jax.lax.stop_gradient(self._target_encoder.apply(train_states.target, o_t))
                    z_t = self._generator.apply(generator_params, x_tm1, a_tm1, x_t)

                    # CALCULATE THE WORLD MODEL LOSS
                    pred_x_t = self._world_model.apply(train_states.world_model.params, z_t, x_tm1, a_tm1)
                    wm_loss = boyl_loss(pred_x_t, x_t).mean()

                    disc_loss = calc_disc_loss(train_states.discriminator.params, x_tm1, a_tm1, z_t)

                    gen_loss = wm_loss - self._config["DISC_IMP_COEF"]* disc_loss
                    return gen_loss
                
                def _discriminator_loss_fn(discriminator_params, traj_batch):
                    a_tm1 = traj_batch.action
                    o_tm1 = traj_batch.obs
                    o_t = traj_batch.next_obs
                    x_tm1 = jax.lax.stop_gradient(self._online_encoder.apply(train_states.online.params, o_tm1))
                    x_t = jax.lax.stop_gradient(self._target_encoder.apply(train_states.target, o_t))
                    z_t = jax.lax.stop_gradient(self._generator.apply(train_states.generator.params, x_tm1, a_tm1, x_t))
                    disc_loss = calc_disc_loss(discriminator_params, x_tm1, a_tm1, z_t)
                    return disc_loss

                # UPDATE THE POLICY NETWORKS
                grad_fn = jax.value_and_grad(_policy_loss_fn, argnums=[0, 1], has_aux=True)
                losses, (online_grads_p, policy_grads) = grad_fn(
                    train_states.online.params, train_states.policy.params,
                    traj_batch, advantages, targets
                )
                new_policy_state = train_states.policy.apply_gradients(grads=policy_grads)  

                # UPDATE THE ONLINE AND WORLD MODEL NETWORKS
                grad_fn = jax.value_and_grad(_wm_loss_fn, argnums=[0, 1])
                wm_loss, (online_grads_w, wm_grads) = grad_fn(
                    train_states.online.params, train_states.world_model.params,
                    traj_batch,
                )
                losses += (wm_loss,)
                # Is there a way to do this in one go?
                online_grads = jax.tree_util.tree_map(lambda x, y: (x + y) / 2, online_grads_p, online_grads_w)
                new_online_state = train_states.online.apply_gradients(grads=online_grads)
                new_wm_state = train_states.world_model.apply_gradients(grads=wm_grads)

                # UPDATE THE GENERATOR
                grad_fn = jax.value_and_grad(_generator_loss_fn)
                gen_loss, gen_grads = grad_fn(
                    train_states.generator.params,
                    traj_batch,
                )
                losses += (gen_loss,)
                new_gen_state = train_states.generator.apply_gradients(grads=gen_grads)

                # UPDATE THE DISCRIMINATOR
                grad_fn = jax.value_and_grad(_discriminator_loss_fn)
                disc_loss, disc_grads = grad_fn(
                    train_states.discriminator.params,
                    traj_batch,
                )
                losses += (disc_loss,)
                new_disc_state = train_states.discriminator.apply_gradients(grads=disc_grads)

                # UPDATE THE TARGET MODEL USING MOVING AVERAGES
                alpha = self._config["TARGET_UPDATE_RATE"]
                new_target_state = jax.tree_util.tree_map(
                    lambda target, online: (
                        1 - alpha
                    ) * target
                    + alpha * online,
                    train_states.target,
                    train_states.online.params,
                )
                # Calculate the distance metrix between the online and target model
                # STEP 1: Flatten both models
                online_params_flat = flatten_params(train_states.online.params)
                target_params_flat = flatten_params(train_states.target)

                # STEP 2: Calculate the distance
                distance = compute_distance(
                    online_params_flat, target_params_flat,
                )
                
                train_states = BOYLTrainState(
                    policy=new_policy_state,
                    online=new_online_state,
                    target=new_target_state,
                    world_model=new_wm_state,
                    generator=new_gen_state,
                    discriminator=new_disc_state,
                )

                return train_states, [losses, distance]

            train_states, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = self._config["MINIBATCH_SIZE"] * self._config["NUM_MINIBATCHES"]
            assert (
                batch_size == self._config["NUM_STEPS"] * self._config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
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
            train_states, metrics = jax.lax.scan(
                _update_minbatch, train_states, minibatches
            )
            return (train_states, traj_batch, advantages, targets, rng), metrics

        update_state = (train_states, traj_batch, advantages, targets, rng)

        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, self._config["UPDATE_EPOCHS"]
        ) 

        (total_loss, (value_loss, actor_loss, entropy_loss), wm_loss, gen_loss, disc_loss), target_dist = metrics
        reward_std = runner_state[-3]
        metric_info = {
            "total_loss": total_loss.mean(),
            "value_loss": value_loss.mean(),
            "actor_loss": actor_loss.mean(),
            "entropy_loss": entropy_loss.mean(),
            "wm_loss": wm_loss.mean(),
            "gen_loss": gen_loss.mean(),
            "disc_loss": disc_loss.mean(),
            "target_dist": target_dist.mean(),
            "reward_std": reward_std,
        }

        train_states = update_state[0]
        step_rewards = traj_batch.info["step_rewards"]
        wm_rewards = traj_batch.info["wm_rewards"]

        episode_rewards = jnp.sum(step_rewards) / jnp.sum(traj_batch.done)
        episode_wm_rewards = jnp.sum(wm_rewards) / jnp.sum(traj_batch.done)

        rng = update_state[-1]
        if self._config.get("DEBUG"):
            def callback(episode_rewards, episode_wm_rewards, metric_info, step):
                print(
                    "Timestep: {}. Episode return: {:.2f}.".format(
                        step, episode_rewards
                    ))
                
                self._logger.write("episode_rewards", episode_rewards, step=step)
                self._logger.write("episode_wm_rewards", episode_wm_rewards, step=step)
                self._logger.write("total_loss", metric_info["total_loss"], step=step)
                self._logger.write("value_loss", metric_info["value_loss"], step=step)
                self._logger.write("actor_loss", metric_info["actor_loss"], step=step)
                self._logger.write("entropy_loss", metric_info["entropy_loss"], step=step)
                self._logger.write("wm_loss", metric_info["wm_loss"], step=step)
                self._logger.write("gen_loss", metric_info["gen_loss"], step=step)
                self._logger.write("disc_loss", metric_info["disc_loss"], step=step)
                self._logger.write("target_distance", metric_info["target_dist"], step=step)
                self._logger.write("reward_std", metric_info["reward_std"], step=step)
            jax.debug.callback(callback, episode_rewards, episode_wm_rewards, metric_info, step)

        runner_state = (train_states, env_state, last_obs, reward_std, rng, step)
        return runner_state

    def run_and_save_gif(self, runner_state, num_steps=1000, output_loc="./logs/Minecraft2D.gif"):
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

        # Set the loger
        self._logger = logger

        # TRAIN LOOP
        num_updates = steps // self._config["NUM_ENVS"] // self._config["NUM_STEPS"]
        update_fn = lambda runner_state, unused: (self._update_step(runner_state), None)
        scan_fn = lambda runner_state: jax.lax.scan(
            update_fn, runner_state, None, length=num_updates
        )
        # scan_fn = jax.jit(chex.assert_max_traces(scan_fn, n=1))
        # runner_state, unused = scan_fn(runner_state)
        runner_state, unused = jax.jit(scan_fn)(runner_state)

        return runner_state