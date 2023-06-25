# Adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_rnn.py
# Please visit the repo above and support the authors.

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from curious_agents.environments.wrappers import LogWrapper, FlattenObservationWrapper


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
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
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # One-hot encode the action
        one_hot_action = jax.nn.one_hot(action, self.action_dim)

        inp = jnp.concatenate([x, one_hot_action], axis=-1)

        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(inp)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer_out)
        layer_out = activation(layer_out)
        layer_out = nn.Dense(x.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            layer_out
        )
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


class PPOAgent():
    def __init__(self, env_name) -> None:
        self._config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": env_name,
        "ANNEAL_LR": False,
        "DEBUG": True,
    }
        self._config["MINIBATCH_SIZE"] = (
            self._config["NUM_ENVS"] * self._config["NUM_STEPS"] // self._config["NUM_MINIBATCHES"]
        )
        self._env, self._env_params = gymnax.make(self._config["ENV_NAME"])
        self._env = FlattenObservationWrapper(self._env)
        self._env = LogWrapper(self._env)

        # INIT NETWORKS
        self._policy_network = ActorCritic(
            self._env.action_space(self._env_params).n, activation=self._config["ACTIVATION"]
        )

        # INIT NETWORK
        self._world_model = WorldModel(
            self._env.action_space(self._env_params).n, activation=self._config["ACTIVATION"]
        )

        # INIT LOGGER
        self._logger = None

    def init_state(self, rng):
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self._config["NUM_MINIBATCHES"] * self._config["UPDATE_EPOCHS"]))
                / self._config["NUM_UPDATES"]
            )
            return self._config["LR"] * frac
        
        #
        rng, _rng = jax.random.split(rng)
        zero_obs = jnp.zeros(self._env.observation_space(self._env_params).shape)
        policy_params = self._policy_network.init(_rng, zero_obs)
        zero_action = jnp.zeros(self._env.action_space(self._env_params).shape, dtype=jnp.int32)
        wm_params = self._world_model.init(_rng, zero_obs, zero_action)

        if self._config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
                optax.adam(self._config["LR"], eps=1e-5),
            )
        policy_train_state = TrainState.create(
            apply_fn=self._policy_network.apply,
            params=policy_params,
            tx=tx,
        )
        wm_train_state = TrainState.create(
            apply_fn=self._world_model.apply,
            params=wm_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self._config["NUM_ENVS"])
        obsv, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(reset_rng, self._env_params)

        return (policy_train_state, wm_train_state, env_state, obsv, rng)
    
    # TRAIN LOOP
    def _update_step(self, use_external_rewards, runner_state, unused):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            policy_train_state, wm_train_state, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self._policy_network.apply(policy_train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, self._config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                self._env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self._env_params)

            # Calcuate the distance between the predicted and the actual observation
            pred_o_t = self._world_model.apply(wm_train_state.params, last_obs, action)
            dist = jnp.linalg.norm(obsv - pred_o_t, axis=-1)


            # Calculate the internal reward
            # TODO: Change this back
            reward =  dist # 0.1*jnp.abs(obsv[..., 1]) + done #dist 
            # jnp.abs(obsv[..., 1]) + done # jnp.square(dist) # - 0.1*jnp.log(jnp.square(dist))
            # jax.debug.print("reward: {x}", x=reward, y=log_prob)
            # reward = use_external_rewards*reward - (1-use_external_rewards)*jnp.log(jnp.square(dist))

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, obsv, info
            )
            runner_state = (policy_train_state, wm_train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, self._config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        policy_train_state, wm_train_state, env_state, last_obs, rng = runner_state
        _, last_val = self._policy_network.apply(policy_train_state.params, last_obs)

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
            def _update_minbatch(train_state, batch_info):
                policy_train_state, wm_train_state = train_state
                traj_batch, advantages, targets = batch_info

                def _agent_loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORKS
                    pi, value = self._policy_network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)
                    
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-self._config["CLIP_EPS"], self._config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)

                    # Maximum??
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
                    entropy_loss = pi.entropy().mean()

                    total_loss = (
                        actor_loss
                        + self._config["VF_COEF"] * value_loss
                        - self._config["ENT_COEF"] * entropy_loss
                    )
                    return total_loss, (value_loss, actor_loss, entropy_loss)

                grad_fn = jax.value_and_grad(_agent_loss_fn, has_aux=True)
                pol_loss, grads = grad_fn(
                    policy_train_state.params, traj_batch, advantages, targets
                )
                policy_train_state = policy_train_state.apply_gradients(grads=grads)

                # UPDATE THE WORLD MODEL
                def _wm_loss_fn(params, traj_batch):
                    # RERUN NETWORK
                    pred_o_t = self._world_model.apply(params, traj_batch.obs, traj_batch.action)
                    # CALCULATE WORLD MODEL LOSS
                    # Don't train on the last step
                    return (jnp.linalg.norm(traj_batch.next_obs - pred_o_t, axis=-1)*(1.0-traj_batch.done)).mean()
                grad_fn = jax.value_and_grad(_wm_loss_fn)
                wm_loss, grads = grad_fn(
                    wm_train_state.params, traj_batch,
                )
                wm_train_state = wm_train_state.apply_gradients(grads=grads)
                return [policy_train_state, wm_train_state], [pol_loss, wm_loss]

            policy_train_state, wm_train_state, traj_batch, advantages, targets, rng = update_state
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
            train_state, loss = jax.lax.scan(
                _update_minbatch, [policy_train_state, wm_train_state], minibatches
            )
 
            update_state = tuple(train_state) + (traj_batch, advantages, targets, rng)
            return update_state, loss

        update_state = (policy_train_state, wm_train_state, traj_batch, advantages, targets, rng)
        update_state, loss = jax.lax.scan(
            _update_epoch, update_state, None, self._config["UPDATE_EPOCHS"]
        )   
      
        (total_loss, (value_loss, actor_loss, entropy_loss)), wm_loss = loss
        loss_info = {
            "total_loss": total_loss.mean(),
            "value_loss": value_loss.mean(),
            "actor_loss": actor_loss.mean(),
            "entropy_loss": entropy_loss.mean(),
            "wm_loss": wm_loss.mean(),
        }
         
        policy_train_state, wm_train_state = update_state[:2]
        metric = traj_batch.info
        rng = update_state[-1]
        if self._config.get("DEBUG"):
            def callback(metric, loss_info):
                return_values = metric["returned_episode_returns"][metric["returned_episode"]]
                timesteps = metric["timestep"][metric["returned_episode"]]

                if len(return_values) > 0:
                    avg_return = np.mean(return_values, axis=0)
                    # for t in range(len(timesteps)):
                    print(f"Timestep: {np.mean(timesteps)}. Episodic return={np.mean(return_values)}")
                    self._logger.write("avg_return", avg_return, step=timesteps[0])
                    self._logger.write("total_loss", loss_info["total_loss"], step=timesteps[0])
                    self._logger.write("value_loss", loss_info["value_loss"], step=timesteps[0])
                    self._logger.write("actor_loss", loss_info["actor_loss"], step=timesteps[0])
                    self._logger.write("entropy_loss", loss_info["entropy_loss"], step=timesteps[0])
                    self._logger.write("wm_loss", loss_info["wm_loss"], step=timesteps[0])
            jax.debug.callback(callback, metric, loss_info)

        runner_state = (policy_train_state, wm_train_state, env_state, last_obs, rng)
        return runner_state, metric["returned_episode_returns"]

    def run(self, runner_state, logger, external_rewards=True, steps=10000, evaluation=False):

        # Set the loger
        self._logger = logger

        # TRAIN LOOP
        num_updates = steps // self._config["NUM_ENVS"] // self._config["NUM_STEPS"]
        update_fn = lambda runner_state, unused: self._update_step(external_rewards, runner_state, unused)
        scan_fn = lambda runner_state: jax.lax.scan(
            update_fn, runner_state, None, length=num_updates
        )
        runner_state, epi_returns = jax.jit(scan_fn)(runner_state)
        return runner_state, epi_returns