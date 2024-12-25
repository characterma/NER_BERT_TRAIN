import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigReader:
    """A class to read and manage YAML configuration files."""

    def __init__(self, config_path: str):
        """
        Initialize the ConfigReader with the path to the YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

    def load_config(self) -> None:
        """Load the configuration from the YAML file."""
        try:
            with self.config_path.open('r') as config_file:
                self.config = yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration.

        Returns:
            Dict[str, Any]: The complete configuration dictionary.
        """
        return self.config

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific value from the configuration.

        Args:
            key (str): The key to look up in the configuration.
            default (Any, optional): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default if not found.
        """
        return self.config.get(key, default)


config_reader = ConfigReader("config.yaml")
config_reader.load_config()

# Get the entire configuration
full_config = config_reader.get_config()
print("Full configuration:", full_config)

