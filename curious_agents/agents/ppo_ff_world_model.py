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
import gymnax
from gymnax.visualize import Visualizer
from curious_agents.environments.wrappers import LogWrapper, FlattenObservationWrapper
from dataclasses import asdict



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

def l2_norm_squared(arr, axis=-1):
    return jnp.sum(jnp.square(arr), axis=axis)

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
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": env_name,
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
        
        rng, policy_rng, wm_rng = jax.random.split(rng, 3)
        zero_obs = jnp.zeros(self._env.observation_space(self._env_params).shape)
        policy_params = self._policy_network.init(policy_rng, zero_obs)
        zero_action = jnp.zeros(self._env.action_space(self._env_params).shape, dtype=jnp.int32)
        wm_params = self._world_model.init(wm_rng, zero_obs, zero_action)

   
        policy_tx = optax.chain(
            optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            optax.adam(self._config["LR"], eps=1e-4),
        )
        wm_tx = optax.chain(
            optax.clip_by_global_norm(self._config["MAX_GRAD_NORM"]),
            optax.adam(self._config["LR"], eps=1e-5),
        )

        policy_train_state = TrainState.create(
            apply_fn=self._policy_network.apply,
            params=policy_params,
            tx=policy_tx,
        )
        wm_train_state = TrainState.create(
            apply_fn=self._world_model.apply,
            params=wm_params,
            tx=wm_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self._config["NUM_ENVS"])
        obsv, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(reset_rng, self._env_params)

        return (policy_train_state, wm_train_state, env_state, obsv, rng)
    
    def _env_step(self, runner_state, unused):
            policy_train_state, wm_train_state, env_state, o_tm1, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self._policy_network.apply(policy_train_state.params, o_tm1)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, self._config["NUM_ENVS"])
            o_t, env_state, original_reward, done, info = jax.vmap(
                self._env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self._env_params)

            # Calcuate the distance between the predicted and the actual observation
            pred_o_t = self._world_model.apply(wm_train_state.params, o_tm1, action)
            reward = l2_norm_squared(o_t - pred_o_t)

            transition = Transition(
                done, action, value, reward, log_prob, o_tm1, o_t, info
            )
            runner_state = (policy_train_state, wm_train_state, env_state, o_t, rng)
            return runner_state, (transition, env_state)

    # TRAIN LOOP
    def _update_step(self, use_external_rewards, runner_state, unused):
        # RUN ENV
        runner_state, (traj_batch, _) = jax.lax.scan(
            self._env_step, runner_state, None, self._config["NUM_STEPS"]
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
                    entropy_loss = - pi.entropy().mean()

                    total_loss = (
                        actor_loss
                        + self._config["VF_COEF"] * value_loss
                        + self._config["ENT_COEF"] * entropy_loss
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
                    # Don't train on the last step. The environment does not provide the step count.
                    # Therefore the world model can not accurately predict the next observation when the 
                    # end of the episode has been reached.
                    # TODO: Potentially add a step count to the environment observation and remove not_last.
                    not_last = 1.0-traj_batch.done
                    return (not_last*l2_norm_squared(traj_batch.next_obs - pred_o_t)).mean()
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
                    step = int(timesteps[0])
                    print(f"Step: {step}. Episode return: {np.mean(return_values)}")
                    self._logger.write("avg_return", avg_return, step=step)
                    self._logger.write("total_loss", loss_info["total_loss"], step=step)
                    self._logger.write("value_loss", loss_info["value_loss"], step=step)
                    self._logger.write("actor_loss", loss_info["actor_loss"], step=step)
                    self._logger.write("entropy_loss", loss_info["entropy_loss"], step=step)
                    self._logger.write("wm_loss", loss_info["wm_loss"], step=step)
            jax.debug.callback(callback, metric, loss_info)

        runner_state = (policy_train_state, wm_train_state, env_state, last_obs, rng)
        return runner_state, metric["returned_episode_returns"]

    def run_and_save_gif(self, runner_state, num_steps=1000, output_loc="./logs/MountainCar.gif"):
        # RUN ENV
        _, (_, env_states) = jax.lax.scan(
            self._env_step, runner_state, None, num_steps
        )

        # Convert to a list
        env_state_arr = env_states.env_state
        print("Converting state..")
        env_state_seq = []
        reward_seq = []
        env_state = type(env_state_arr)
        dict_env_arr = asdict(env_state_arr)
        fields = env_state.__annotations__.keys()
        
        for i in range(0, num_steps):
            entry_dict = {}
            # Calculate the first episode's information
            for key in fields:
                entry_dict[key] = dict_env_arr[key][i][0]
            env_state_seq.append(env_state(**entry_dict))
            reward_seq.append(env_states.episode_returns[i][0])

        # Create a gif that visualises the experience.
        print("Starting the saving process..")
        from pyvirtualdisplay import Display  # type: ignore
        display = Display(visible=0, size=(1400, 900))
        display.start()
        print("Saving gif..")
        vis = Visualizer(self._env, self._env_params, env_state_seq, reward_seq)
        vis.animate(output_loc, view=False)

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