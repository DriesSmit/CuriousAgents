# Adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_rnn.py
# Please visit the repo above and support the authors.

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
from curious_agents.environments.wrappers import FlattenObservationWrapper, LogWrapper


class ScannedRNN(nn.Module):
    """RNN Architecture"""
    # nn.scan function processes sequences of data
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """
        Applies the module (GRU cell).

        Args:
            carry: RNN hidden state
            x: Input sequences to be split into inputs and reset signals

        Returns:
            _description_
        """
        # Initialise RNN state to hidden state, obtain inputs and reset signals
        rnn_state = carry
        ins, resets = x
        
        # TODO: INVESTIGATE RESETS
        # Where resets are (?), execute initialize_carry fn, else keep rnn_state
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """
        Initialise GRU hidden state

        Args:
            batch_size: TODO
            hidden_size: TODO

        Returns:
            TODO
        """
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        """
        Generate policy over actions and compute value of current state.
        Incorporates RNN to maintain memory of past interactions

        Args:
            hidden: Hidden state of RNN
            x: Input - observations and done signals

        Returns:
            TODO
        """
        # Extract observations and done signals from input
        obs, dones = x
        
        # Create embedding from observations
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # Embedding is passed into RNN, hidden state and embedding is updated
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # Actor - Dense layers applied to updated embedding
        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Actor - categorical distribution represents policy
        pi = distrax.Categorical(logits=actor_mean)

        # Critic - Dense layers applied to updated embedding
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # Return updated hidden state, policy and state value
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """Contains information for an environment transition"""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

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
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": env_name,
        "ANNEAL_LR": True,
        "DEBUG": True,
        }
        self._config["MINIBATCH_SIZE"] = (
            self._config["NUM_ENVS"] * self._config["NUM_STEPS"] // self._config["NUM_MINIBATCHES"]
        )
        
        # Create environment
        self._env, self._env_params = gymnax.make(self._config["ENV_NAME"])
        self._env = FlattenObservationWrapper(self._env)
        self._env = LogWrapper(self._env)

        # INIT NETWORK
        # Policy Network - Actor Critic network for hidden state, policy and state value
        self._network = ActorCriticRNN(self._env.action_space(self._env_params).n, config=self._config)
        

    def init_state(self, rng):
        """Initialise agent state using PRNG key"""
        def linear_schedule(count):
            # frac = (
            #     1.0
            #     - (count // (self._config["NUM_MINIBATCHES"] * self._config["UPDATE_EPOCHS"]))
            #     / self._config["NUM_UPDATES"]
            # )
            frac = 1.0
            return self._config["LR"] * frac
        
        # Split provided PRNG key
        rng, _rng = jax.random.split(rng)
        
        # INIT POLICY NETWORK
        init_x = (
            jnp.zeros(
                (1, self._config["NUM_ENVS"], *self._env.observation_space(self._env_params).shape)
            ),
            jnp.zeros((1, self._config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(self._config["NUM_ENVS"], 128)
        network_params = self._network.init(_rng, init_hstate, init_x)
        
        # Define optimizers for regular and annealing sequence config
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
            
        # Define training state for policy network
        train_state = TrainState.create(
            apply_fn=self._network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV - Reset environment to initial state specified by PRNG key
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self._config["NUM_ENVS"])
        obsv, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(reset_rng, self._env_params)
        
        # TODO Remove? This line seems redundant - init_state defined earlier
        # is never modified?
        init_hstate = ScannedRNN.initialize_carry(self._config["NUM_ENVS"], 128)

        # Initialise runner state and return
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((self._config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )

        return runner_state
    
    def _update_step(self, runner_state, unused):
        """Main training loop"""
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            """Defines agent interaction with environment for a single timestep"""
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # Obtain policy and state value from policy (Actor Critic) network
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = self._network.apply(train_state.params, hstate, ac_in)
            
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # PERFORM STEP IN ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, self._config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                self._env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, self._env_params)
            
            # Update transition information
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )
            
            # Step once and update state
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        # Obtain trajectories
        initial_hstate = runner_state[-2]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, self._config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, _, last_val = self._network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        def _calculate_gae(traj_batch, last_val):
            """
            Generalized Advantage Estimation

            Args:
                traj_batch: Batch of sampled trajectories
                last_val: Value for last observation obtained earlier
                
            Return:
                Calculated advantages and target values (advantages + values)
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
                    + self._config["GAMMA"] * self._config["GAE_LAMBDA"] * (1 - done) * gae
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
            def _update_minbatch(train_state, batch_info):
                """
                Use minibatch to perform updates to agent models

                Args:
                    train_states: TODO
                    batch_info: TODO

                Returns:
                    Training states, policy loss, world model loss, and
                    distance metric between online and target model params
                """
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    """
                    Calculate Agent Loss

                    Args:
                        params: Policy network parameters
                        init_hstate: Hidden state of RNN
                        traj_batch: Sampled trajectories batch
                        gae: Advantages from _calculate_gae()
                        targets: Targets from _calculate_gae()

                    Returns:
                        Total loss - weighted sum of actor, value and entropy losses
                    """
                    # RERUN NETWORK
                    _, pi, value = self._network.apply(
                        params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                    )
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
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - self._config["CLIP_EPS"],
                            1.0 + self._config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    
                    # Calculate entropy loss - penalize deterministic policies
                    entropy = pi.entropy().mean()

                    # Calculate total loss - weighted sum of actor, entropy losses
                    total_loss = (
                        loss_actor
                        + self._config["VF_COEF"] * value_loss
                        - self._config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                # UPDATE THE POLICY - Compute gradients w.r.t policy parameters
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            # Extract and update information from update state
            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state

            # Reshape and divide trajectories into minibatches
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, self._config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], self._config["NUM_MINIBATCHES"], -1] 
                                + list(x.shape[2:]),), 1, 0,),
                shuffled_batch,
            )

            # Apply update minibatch function to every minibatch
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        # Create update state and perform actor models update
        init_hstate = initial_hstate[None, :]  # TBH
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, self._config["UPDATE_EPOCHS"]
        )
        
        # Obtain current training state, extract metrics after update conclusion
        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]
        
        # Debugging functionality - log metrics for each step
        if self._config.get("DEBUG"):

            def callback(info):
                return_values = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                timesteps = (
                    info["timestep"][info["returned_episode"]] * self._config["NUM_ENVS"]
                )
                for t in range(len(timesteps)):
                    print(
                        f"global step={timesteps[t]}, episodic return={return_values[t]}"
                    )

            jax.debug.callback(callback, metric)

        # Update runner state with newest information and return
        runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        return runner_state, metric["returned_episode_returns"]

    def run(self, runner_state, external_rewards=True, steps=10000, evaluation=False):
        """
        Execute agent training loop

        Args:
            runner_state: State of agent at current timestep
            external_rewards: Whether to use external rewards (default: {True})
            steps: Number of training steps (default: {10000})
            evaluation: Whether the following is an evaluation run (default: {False})

        Returns:
            The final state of the agent
        """
        # TRAIN LOOP
        # 5e5 steps
        if not external_rewards:
            return runner_state, 0.0
        
        num_updates = steps // self._config["NUM_ENVS"] // self._config["NUM_STEPS"]
        scan_fn = lambda runner_state: jax.lax.scan(
            self._update_step, runner_state, None, length=num_updates
        )
        runner_state, epi_returns = jax.jit(scan_fn)(runner_state)
        return runner_state, epi_returns