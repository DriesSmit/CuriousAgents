from curious_agents.agents.ppo_ff import PPOAgent as FFPPOAgent
from curious_agents.agents.ppo_ff_world_model import PPOAgent as WorldModelPPOAgent
from curious_agents.agents.ppo_byol_explore import PPOAgent as BYOLPPOAgent
from curious_agents.agents.ppo_byol_hindsight import PPOAgent as BYOLHindsightPPOAgent
from omegaconf import DictConfig

def create_agent(cfg: DictConfig):
    if cfg.agent.name == "ppo_ff":
        return FFPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_ff_world_model":
        return WorldModelPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_byol_explore":
        return BYOLPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_byol_hindsight":
        return BYOLHindsightPPOAgent(cfg.env_name)
    

    
    raise ValueError(f"Unknown agent: {cfg.agent.name}")