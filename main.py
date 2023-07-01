import hydra
from omegaconf import DictConfig
from curious_agents.parameter_manager import ParameterManager
from curious_agents.agents import create_agent
from curious_agents.loggers.tensorboard import TensorBoardLogger
import jax


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # Setup the logger
    logger = TensorBoardLogger(cfg.log_dir)

    # Create the agent
    agent = create_agent(cfg)

    # For saving and loading of the runner state
    manager = ParameterManager(cfg.log_dir)
    
    # Optionally load the runner state
    rng = jax.random.PRNGKey(cfg.seed)
    runner_state = agent.init_state(rng)

    if cfg.load_state:
        # Load runner state
        runner_state = manager.load(runner_state)

    if cfg.train:
        # Run the agent in exploration mode
        runner_state = agent.run(runner_state, logger, steps=cfg.training_steps)
        
        # Save runner state
        manager.save(runner_state)

    if cfg.visualise:
        # Visialise the Gymnax environment
        agent.run_and_save_gif(runner_state)

if __name__ == "__main__":
    main()