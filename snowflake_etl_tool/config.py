import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from dotenv import load_dotenv

class Config:
    """Load configuration from YAML file and .env environment."""

    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(env_file)
        self._config_path = Path(config_path)
        self.config_data: Dict = {}
        if self._config_path.exists():
            with self._config_path.open() as f:
                self.config_data = yaml.safe_load(f) or {}
        self._load_env()

    def _load_env(self) -> None:
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.snowflake_user = os.getenv("SNOWFLAKE_USER")
        self.snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
        self.snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.snowflake_role = os.getenv("SNOWFLAKE_ROLE", "SYSADMIN")

    @property
    def export_schema_dir(self) -> Optional[str]:
        return self.config_data.get("export_schema_dir")

    @property
    def transformations(self) -> List[str]:
        return self.config_data.get("transformations", [])

    @property
    def target_database(self) -> str:
        return self.config_data.get("target_database", "AUTO_ETL_DB")

    @property
    def target_schema(self) -> str:
        return self.config_data.get("target_schema", "PUBLIC")

    @property
    def data_sources(self) -> List[Dict]:
        return self.config_data.get("data_sources", [])
