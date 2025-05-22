#!/usr/bin/env python3
"""
Verify that the extracted SQL files match the original definitions
"""
import re
import os

def extract_staging_sql_statements(file_path):
    """Extract staging SQL statements from Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all CREATE TABLE statements for staging tables
    statements = {}
    for match in re.finditer(r'CREATE OR REPLACE TABLE STAGING_(\w+)\s*\((.*?)\);', 
                           content, re.DOTALL):
        table_name = match.group(1).lower()
        full_statement = match.group(0).strip()
        statements[table_name] = full_statement
    
    return statements

def extract_dimension_sql_statements(file_path):
    """Extract dimension SQL statements from Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all CREATE TABLE statements for dimension tables
    statements = {}
    for match in re.finditer(r'CREATE OR REPLACE TABLE Dim_(\w+)\s*\((.*?)\)', 
                           content, re.DOTALL):
        table_name = match.group(1).lower()
        full_statement = match.group(0).strip()
        statements[table_name] = full_statement
    
    return statements

def extract_fact_sql_statements(file_path):
    """Extract fact SQL statements from Python file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all CREATE TABLE statements for fact tables
    statements = {}
    for match in re.finditer(r'CREATE OR REPLACE TABLE Fact_(\w+)\s*\((.*?)\)', 
                           content, re.DOTALL):
        table_name = match.group(1).lower()
        full_statement = match.group(0).strip()
        statements[table_name] = full_statement
    
    return statements

def clean_sql(sql):
    """Clean SQL for comparison by removing extra whitespace"""
    # Remove multiple spaces
    sql = re.sub(r'\s+', ' ', sql)
    # Remove spaces around parentheses and commas
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    sql = re.sub(r'\s*,\s*', ',', sql)
    return sql.strip()

def verify_staging_sql_files():
    """Verify that the extracted staging SQL files match the original code"""
    print("\n=== Verifying Staging Tables ===")
    # Extract SQL statements from Python file
    original_statements = extract_staging_sql_statements('rahil/create_tables.py')
    
    # Check each SQL file
    path = 'private_ddl/rahil'
    all_match = True
    
    for filename in os.listdir(path):
        if not filename.startswith('staging_') or not filename.endswith('.sql'):
            continue
        
        table_name = filename.replace('staging_', '').replace('.sql', '')
        file_path = os.path.join(path, filename)
        
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Extract just the CREATE TABLE part from the file content
        create_match = re.search(r'CREATE OR REPLACE TABLE.*?;', file_content, re.DOTALL)
        if not create_match:
            print(f"❌ Error: No CREATE TABLE statement found in {filename}")
            all_match = False
            continue
            
        file_sql = create_match.group(0)
        
        # Compare with original statement (ignoring whitespace differences)
        if table_name in original_statements:
            original_sql = original_statements[table_name]
            
            if clean_sql(file_sql) == clean_sql(original_sql):
                print(f"✅ {filename} matches original definition")
            else:
                print(f"❌ {filename} DOES NOT match original definition")
                all_match = False
                print(f"  Original: {clean_sql(original_sql)}")
                print(f"  File:     {clean_sql(file_sql)}")
        else:
            print(f"❓ {filename} does not correspond to any original table definition")
            all_match = False
    
    if all_match:
        print("\n✅ All staging SQL files match the original definitions!")
    else:
        print("\n❌ Some staging SQL files do not match the original definitions.")
    
    return all_match

def verify_dimension_sql_files():
    """Verify that the extracted dimension SQL files match the original code"""
    print("\n=== Verifying Dimension Tables ===")
    # Extract SQL statements from Python file
    original_statements = extract_dimension_sql_statements('rahil/create_dimension_tables.py')
    
    # Check each SQL file
    path = 'private_ddl/rahil'
    all_match = True
    
    for filename in os.listdir(path):
        if not filename.startswith('dim_') or not filename.endswith('.sql'):
            continue
        
        table_name = filename.replace('dim_', '').replace('.sql', '')
        file_path = os.path.join(path, filename)
        
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Extract just the CREATE TABLE part from the file content
        create_match = re.search(r'CREATE OR REPLACE TABLE.*?\)', file_content, re.DOTALL)
        if not create_match:
            print(f"❌ Error: No CREATE TABLE statement found in {filename}")
            all_match = False
            continue
            
        file_sql = create_match.group(0)
        
        # Compare with original statement (ignoring whitespace differences)
        if table_name in original_statements:
            original_sql = original_statements[table_name]
            
            if clean_sql(file_sql) == clean_sql(original_sql):
                print(f"✅ {filename} matches original definition")
            else:
                print(f"❌ {filename} DOES NOT match original definition")
                all_match = False
                print(f"  Original: {clean_sql(original_sql)}")
                print(f"  File:     {clean_sql(file_sql)}")
        else:
            print(f"❓ {filename} does not correspond to any original table definition")
            all_match = False
    
    if all_match:
        print("\n✅ All dimension SQL files match the original definitions!")
    else:
        print("\n❌ Some dimension SQL files do not match the original definitions.")
    
    return all_match

def verify_fact_sql_files():
    """Verify that the extracted fact SQL files match the original code"""
    print("\n=== Verifying Fact Tables ===")
    # Extract SQL statements from Python file
    original_statements = extract_fact_sql_statements('rahil/create_fact_tables.py')
    
    # Check each SQL file
    path = 'private_ddl/rahil'
    all_match = True
    
    for filename in os.listdir(path):
        if not filename.startswith('fact_') or not filename.endswith('.sql'):
            continue
        
        table_name = filename.replace('fact_', '').replace('.sql', '')
        file_path = os.path.join(path, filename)
        
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Extract just the CREATE TABLE part from the file content
        create_match = re.search(r'CREATE OR REPLACE TABLE.*?\)', file_content, re.DOTALL)
        if not create_match:
            print(f"❌ Error: No CREATE TABLE statement found in {filename}")
            all_match = False
            continue
            
        file_sql = create_match.group(0)
        
        # Compare with original statement (ignoring whitespace differences)
        if table_name in original_statements:
            original_sql = original_statements[table_name]
            
            if clean_sql(file_sql) == clean_sql(original_sql):
                print(f"✅ {filename} matches original definition")
            else:
                print(f"❌ {filename} DOES NOT match original definition")
                all_match = False
                print(f"  Original: {clean_sql(original_sql)}")
                print(f"  File:     {clean_sql(file_sql)}")
        else:
            print(f"❓ {filename} does not correspond to any original table definition")
            all_match = False
    
    if all_match:
        print("\n✅ All fact SQL files match the original definitions!")
    else:
        print("\n❌ Some fact SQL files do not match the original definitions.")
    
    return all_match

def verify_sql_files():
    """Verify that all extracted SQL files match the original code"""
    staging_ok = verify_staging_sql_files()
    dimension_ok = verify_dimension_sql_files()
    fact_ok = verify_fact_sql_files()
    
    if staging_ok and dimension_ok and fact_ok:
        print("\n✅ SUMMARY: All SQL files match the original definitions!")
    else:
        print("\n❌ SUMMARY: Some SQL files do not match the original definitions.")

if __name__ == "__main__":
    verify_sql_files() 