from snowflake_etl_tool.schema_inference import infer_schema_csv
from pathlib import Path


def test_infer_schema_csv():
    path = Path(__file__).parent / 'data' / 'dummy.csv'
    schema = infer_schema_csv(path)
    assert schema[0]['type'] == 'INTEGER'
    assert schema[1]['type'].startswith('VARCHAR')
    assert schema[2]['type'] == 'FLOAT'
