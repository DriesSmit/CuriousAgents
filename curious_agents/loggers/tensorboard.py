import os
# Write the config to the log directory.
def write_config_to_logs(cfg: OmegaConf, log_dir: str) -> None:

    # Write the config to the log directory.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    OmegaConf.save(cfg, join(log_dir, "config.yaml"))