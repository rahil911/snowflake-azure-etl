from snowflake_etl_tool.ddl_generator import create_table_sql


def test_create_table_sql():
    schema = [
        {'name': 'id', 'type': 'INTEGER', 'nullable': False},
        {'name': 'name', 'type': 'TEXT', 'nullable': True},
    ]
    sql = create_table_sql('TEST', schema)
    assert 'CREATE OR REPLACE TABLE TEST' in sql
    assert 'id INTEGER NOT NULL' in sql
    assert 'name TEXT' in sql
