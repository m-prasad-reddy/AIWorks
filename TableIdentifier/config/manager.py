# config/manager.py
import os
import json
from typing import Dict

class DBConfigManager:
    def load_configs(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path) as f:
            configs = json.load(f)
        
        self._validate_configs(configs)
        return configs

    def _validate_configs(self, configs: Dict):
        if not isinstance(configs, dict):
            raise ValueError("Config file should contain a dictionary of configurations")
        
        required_keys = {'server', 'database', 'username', 'password', 'driver'}
        for key, config in configs.items():
            if not isinstance(config, dict):
                raise ValueError(f"Configuration for {key} must be a dictionary")
            if not required_keys.issubset(config.keys()):
                missing = required_keys - set(config.keys())
                raise ValueError(f"Missing keys in {key} config: {', '.join(missing)}")