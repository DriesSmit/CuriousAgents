from curious_agents.agents.ppo_world_model import PPOAgent
from omegaconf import DictConfig

def create_agent(cfg: DictConfig) -> PPOAgent:
    if cfg.agent.name == "ppo_rnn":
        return PPOAgent(cfg.env_name)
    
    raise ValueError(f"Unknown agent: {cfg.name}")