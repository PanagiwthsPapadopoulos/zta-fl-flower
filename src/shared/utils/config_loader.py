import os
import yaml


def load_yaml_configs() -> dict:
    config_dir = os.environ.get("CONFIG_PATH")
    if not config_dir and os.path.exists("/app/config"):
        config_dir = "/app/config"
    if not config_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(base_dir, "config")

    master_config = {}
    if not os.path.exists(config_dir):
        print(f"⚠️ ERROR: Configuration directory not found at {config_dir}")
        return master_config

    files = ["network.yaml", "training.yaml", "security.yaml", "threat.yaml", "admin.yaml"]
    for file in files:
        path = os.path.join(config_dir, file)
        if os.path.exists(path):
            with open(path, "r") as f:
                master_config.update(yaml.safe_load(f) or {})
    return master_config


def get_merged_config(flower_run_config: dict) -> dict:
    """Merges the native Flower configuration with our YAML Single Source of Truth (SSOT)."""
    return {**flower_run_config, **load_yaml_configs()}