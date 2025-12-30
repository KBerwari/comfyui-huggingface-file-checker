"""
Configuration management for HuggingFace File Checker.
Stores user settings like local directories in a JSON config file.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Use standard locations
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home()))
    else:  # Linux/Mac
        base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    
    config_dir = base / 'hf-file-checker'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / 'config.json'


@dataclass
class DirectoryConfig:
    """Configuration for a single local directory."""
    path: str
    name: str = ""  # Optional friendly name
    scan_mode: str = "metadata"  # "metadata" (JSON files) or "files" (direct hash)
    extensions: List[str] = field(default_factory=lambda: ['.safetensors', '.ckpt', '.pt', '.bin'])
    enabled: bool = True
    added: str = ""  # ISO timestamp
    
    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).name
        if not self.added:
            self.added = datetime.now().isoformat()


@dataclass 
class ServerConfig:
    """Server configuration."""
    port: int = 7860
    host: str = "127.0.0.1"
    auto_scan_on_start: bool = True


@dataclass
class Config:
    """Main configuration."""
    directories: List[DirectoryConfig] = field(default_factory=list)
    server: ServerConfig = field(default_factory=ServerConfig)
    version: int = 2 
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'server': asdict(self.server),
            'directories': [asdict(d) for d in self.directories]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        config.version = data.get('version', 1)
        
        if 'server' in data:
            config.server = ServerConfig(**data['server'])
        
        if 'directories' in data:
            config.directories = [
                DirectoryConfig(**d) for d in data['directories']
            ]
        
        return config


class ConfigManager:
    """Manages loading and saving configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or get_config_path()
        self._config: Optional[Config] = None
    
    @property
    def config(self) -> Config:
        """Get current config, loading from disk if needed."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self) -> Config:
        """Load config from disk."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return Config.from_dict(data)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
        
        return Config()
    
    def save(self) -> bool:
        """Save config to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def add_directory(
        self, 
        path: str, 
        name: str = "",
        scan_mode: str = "files",
        extensions: Optional[List[str]] = None
    ) -> DirectoryConfig:
        """Add a new directory to the config."""
        # Normalize path
        path = str(Path(path).resolve())
        
        # Check if already exists
        for d in self.config.directories:
            if Path(d.path).resolve() == Path(path).resolve():
                # Update existing
                d.name = name or d.name
                d.scan_mode = scan_mode
                if extensions:
                    d.extensions = extensions
                d.enabled = True
                self.save()
                return d
        
        # Create new
        dir_config = DirectoryConfig(
            path=path,
            name=name,
            scan_mode=scan_mode,
            extensions=extensions or ['.safetensors', '.ckpt', '.pt', '.bin']
        )
        self.config.directories.append(dir_config)
        self.save()
        return dir_config
    
    def remove_directory(self, path: str) -> bool:
        """Remove a directory from the config."""
        path = str(Path(path).resolve())
        original_len = len(self.config.directories)
        self.config.directories = [
            d for d in self.config.directories 
            if Path(d.path).resolve() != Path(path).resolve()
        ]
        if len(self.config.directories) < original_len:
            self.save()
            return True
        return False
    
    def get_enabled_directories(self) -> List[DirectoryConfig]:
        """Get all enabled directories."""
        return [d for d in self.config.directories if d.enabled]
    
    def set_directory_enabled(self, path: str, enabled: bool) -> bool:
        """Enable or disable a directory."""
        path = str(Path(path).resolve())
        for d in self.config.directories:
            if Path(d.path).resolve() == Path(path).resolve():
                d.enabled = enabled
                self.save()
                return True
        return False


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
