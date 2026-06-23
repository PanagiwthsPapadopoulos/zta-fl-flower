import os
import yaml

def load_yaml_configs() -> dict:
    """Loads and merges all YAML configurations into a single dictionary."""
    
    # Prioritize explicitly passed environment variables
    config_dir = os.environ.get("CONFIG_PATH")
    
    # Primary Fallback: The known Docker container path
    if not config_dir and os.path.exists("/app/config"):
        config_dir = "/app/config"
        
    # Secondary Fallback: File-relative path (for local Mac/Linux testing outside Docker)
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
                file_config = yaml.safe_load(f) or {}
                master_config.update(file_config)
        else:
            print(f"⚠️ Warning: Config file {file} not found in {config_dir}")
            
    return master_config

def get_merged_config(flower_run_config: dict) -> dict:
    """
    Merges the native Flower configuration with our YAML Single Source of Truth (SSOT).
    YAML configurations will explicitly OVERRIDE Flower TOML variables to prevent split-brain logic.
    """
    app_config = load_yaml_configs()
    
    # YAML (app_config) has absolute authority over TOML
    return {**flower_run_config, **app_config}