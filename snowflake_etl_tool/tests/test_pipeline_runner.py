from snowflake_etl_tool.pipeline_runner import run_pipeline
from snowflake_etl_tool.connection import SnowflakeConnection
import yaml
from pathlib import Path


commands = []


class DummyConn(SnowflakeConnection):
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, sql):
        commands.append(sql)
        return []

    def close(self):
        pass


def test_run_pipeline(monkeypatch, tmp_path):
    # Use dummy connection
    def dummy_conn(config):
        return DummyConn()

    monkeypatch.setattr('snowflake_etl_tool.pipeline_runner.SnowflakeConnection', dummy_conn)
    # Build temporary config with schema export and transformation
    data_path = Path(__file__).parent / 'data' / 'dummy.csv'
    transform = tmp_path / 'trans.sql'
    transform.write_text('SELECT 1;')
    config = {
        'target_database': 'TESTDB',
        'export_schema_dir': str(tmp_path),
        'transformations': [str(transform)],
        'data_sources': [
            {'name': 'dummy', 'path': str(data_path), 'file_format': 'csv'}
        ],
    }
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text(yaml.safe_dump(config))
    run_pipeline(str(cfg_file))
    assert any('CREATE DATABASE IF NOT EXISTS TESTDB' in c for c in commands)
    assert (tmp_path / 'dummy_schema.yaml').exists()
    assert 'SELECT 1;' in commands[-1]
