from curious_agents.agents.ppo_rnn import PPOAgent
from curious_agents.loggers.terminal import Logger
from jax import jit
def main() -> None:
    # Hyperparameters
    explore_first = False
    pre_training_steps = 100000
    fine_tuning_steps = 10000

    # Setup the environment
    env_name = "CartPole-v1"

    # Setup the logger
    logger = Logger()

    # Setup the agent
    agent = PPOAgent(env_name)
    agent_state = agent.init_state()
    run_fn = jit(agent.run)

    if explore_first:
        # Run the agent in exploration mode
        agent_state, log_info = run_fn(agent_state, external_rewards=False,
                                steps=pre_training_steps,)
        logger.write(log_info, "curiousity_driven")

    # Run the agent in fine-tuning mode
    agent_state, log_info = run_fn(agent_state, external_rewards=False,
                        steps=fine_tuning_steps,)
    logger.write(log_info, "explore_with_external_rewards")

    # Evaluate the agent's final performance.
    agent_state, log_info = run_fn(agent_state, external_rewards=False,
                                        steps=10000, evaluation=True)
    logger.write(log_info, "final_evaluation")
    


if __name__ == "__main__":
    main()