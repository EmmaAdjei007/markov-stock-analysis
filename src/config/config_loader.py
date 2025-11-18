"""
Configuration loader with YAML file support and environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """
    Configuration manager that loads settings from YAML and environment variables.

    Supports nested configuration access via dot notation or dictionary access.
    Environment variables take precedence over YAML configuration.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Determine config path
        if config_path is None:
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Checks environment variables first, then YAML config.
        Environment variable names are uppercase with underscores.

        Examples:
            config.get('app.name') -> checks APP_NAME env var, then config['app']['name']
            config.get('markov.default_n_states') -> MARKOV_DEFAULT_N_STATES or config value

        Args:
            key: Configuration key in dot notation (e.g., 'app.name')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Check environment variable first (uppercase with underscores)
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)

        # Navigate through nested config
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (bool, int, float, or string)
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Numeric
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Top-level section name (e.g., 'markov', 'simulation')

        Returns:
            Dictionary containing section configuration
        """
        return self._config.get(section, {})

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None

    # Convenience properties for common config sections

    @property
    def app(self) -> Dict[str, Any]:
        """Application settings."""
        return self.get_section('app')

    @property
    def data(self) -> Dict[str, Any]:
        """Data management settings."""
        return self.get_section('data')

    @property
    def markov(self) -> Dict[str, Any]:
        """Markov model settings."""
        return self.get_section('markov')

    @property
    def simulation(self) -> Dict[str, Any]:
        """Simulation settings."""
        return self.get_section('simulation')

    @property
    def backtesting(self) -> Dict[str, Any]:
        """Backtesting settings."""
        return self.get_section('backtesting')

    @property
    def portfolio(self) -> Dict[str, Any]:
        """Portfolio settings."""
        return self.get_section('portfolio')

    @property
    def visualization(self) -> Dict[str, Any]:
        """Visualization settings."""
        return self.get_section('visualization')

    @property
    def ai(self) -> Dict[str, Any]:
        """AI model settings."""
        return self.get_section('ai')

    @property
    def dashboard(self) -> Dict[str, Any]:
        """Dashboard settings."""
        return self.get_section('dashboard')

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Logging configuration."""
        return self.get_section('logging')

    # API Key helpers

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for specified provider.

        Checks environment variables and Streamlit secrets.

        Args:
            provider: API provider name ('openai', 'gemini', 'groq')

        Returns:
            API key or None if not found
        """
        env_var = f"{provider.upper()}_API_KEY"

        # Check environment variable
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

        # Check Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and env_var in st.secrets:
                return st.secrets[env_var]
        except (ImportError, FileNotFoundError):
            pass

        return None

    def has_api_key(self, provider: str) -> bool:
        """
        Check if API key exists for provider.

        Args:
            provider: API provider name

        Returns:
            True if API key is configured
        """
        return self.get_api_key(provider) is not None


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Optional custom configuration file path
        reload: Force reload configuration from file

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config(config_path)

    return _config_instance


# Example usage:
if __name__ == "__main__":
    config = get_config()

    print("App Name:", config.get('app.name'))
    print("Default States:", config.get('markov.default_n_states'))
    print("Simulation Days:", config.get('simulation.default_n_days'))
    print("AI Provider:", config.get('ai.provider'))

    # Check API keys
    print("\nAPI Keys configured:")
    for provider in ['openai', 'gemini', 'groq']:
        status = "✓" if config.has_api_key(provider) else "✗"
        print(f"  {provider}: {status}")
