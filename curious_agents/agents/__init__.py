from curious_agents.agents.ppo_ff import PPOAgent as FFPPOAgent
from curious_agents.agents.ppo_ff_world_model import PPOAgent as WorldModelPPOAgent
from curious_agents.agents.ppo_boyl_explore import PPOAgent as BOYLPPOAgent
from curious_agents.agents.ppo_boyl_hindsight import PPOAgent as BOYLHindsightPPOAgent
from omegaconf import DictConfig

def create_agent(cfg: DictConfig):
    if cfg.agent.name == "ppo_ff":
        return FFPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_ff_world_model":
        return WorldModelPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_boyl_explore":
        return BOYLPPOAgent(cfg.env_name)
    elif cfg.agent.name == "ppo_boyl_hindsight":
        return BOYLHindsightPPOAgent(cfg.env_name)
    

    
    raise ValueError(f"Unknown agent: {cfg.agent.name}")