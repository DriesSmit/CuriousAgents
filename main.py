import hydra
from omegaconf import DictConfig
from curious_agents.agents import create_agent
from curious_agents.loggers.tensorboard import TensorBoardLogger
import jax

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # Setup the logger
    logger = TensorBoardLogger("./logs")

    # Create the agent
    agent = create_agent(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    runner_state = agent.init_state(rng)

    if cfg.explore_first:
        # Run the agent in exploration mode
        runner_state, log_info = agent.run(runner_state, logger, steps=cfg.pre_training_steps, external_rewards=False,)

        print("log_info: ", log_info)
        exit()
        

    # Run the agent in fine-tuning mode
    # runner_state, log_info = agent.run(runner_state, steps=cfg.fine_tuning_steps, external_rewards=True)
    # logger.write(log_info, "explore_with_external_rewards")

    # Evaluate the agent's final performance.
    # runner_state, log_info = agent.run(runner_state, external_rewards=False,
    #                                     steps=10000, evaluation=True)
    # logger.write(log_info, "final_evaluation")

if __name__ == "__main__":
    main()