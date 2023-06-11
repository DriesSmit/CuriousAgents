from curious_agents.agents.ppo_rnn import PPOAgent
from curious_agents.loggers.terminal import Logger
import jax
def main() -> None:
    # Hyperparameters
    explore_first = False
    pre_training_steps = 100000
    fine_tuning_steps = 100000
    seed = 1234

    # Setup the environment
    env_name = "CartPole-v1"

    # Setup the logger
    logger = Logger()

    # Setup the agent
    agent = PPOAgent(env_name)
    rng = jax.random.PRNGKey(seed)
    runner_state = agent.init_state(rng)

    if explore_first:
        # Run the agent in exploration mode
        runner_state, log_info = agent.run(runner_state, steps=pre_training_steps, external_rewards=False,)
        logger.write(log_info, "curiousity_driven")

    # Run the agent in fine-tuning mode
    runner_state, log_info = agent.run(runner_state, steps=fine_tuning_steps, external_rewards=True)
    logger.write(log_info, "explore_with_external_rewards")

    # Evaluate the agent's final performance.
    # runner_state, log_info = agent.run(runner_state, external_rewards=False,
    #                                     steps=10000, evaluation=True)
    # logger.write(log_info, "final_evaluation")
    


if __name__ == "__main__":
    main()