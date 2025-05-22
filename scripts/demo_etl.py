#!/usr/bin/env python3
"""
Demo Script for Snowflake ETL Project

This script demonstrates the end-to-end ETL process:
1. Creates a temporary .env file from environment variables
2. Initializes the project with the CLI scaffolder
3. Executes the ETL process
4. Shows row counts for all tables

Usage:
    SNOWFLAKE_ACCOUNT=account SNOWFLAKE_USER=user SNOWFLAKE_PASSWORD=password python3 scripts/demo_etl.py
"""
import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def print_step(step_num, title):
    """Print a formatted step header"""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: {title}")
    print("=" * 80)

def get_credentials_from_env():
    """Get Snowflake credentials from environment variables"""
    print_step(1, "LOADING CREDENTIALS FROM ENVIRONMENT")
    
    # Required environment variables
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Please set these environment variables before running the script.")
        sys.exit(1)
    
    # Create credentials dictionary
    credentials = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_PASSWORD"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        "azure_storage": os.environ.get("AZURE_STORAGE_ACCOUNT", "sp72storage.blob.core.windows.net"),
        "user_name": os.environ.get("USER_NAME", os.environ.get("USER", "demo_user"))
    }
    
    # Print what we're using (except password)
    for k, v in credentials.items():
        if k != "password":
            print(f"Using {k}: {v}")
    
    return credentials

def create_temp_env_file(credentials, temp_dir):
    """Create a temporary .env file"""
    print_step(2, "CREATE ENVIRONMENT FILE")
    
    env_content = f"""# Snowflake credentials
SNOWFLAKE_ACCOUNT={credentials['account']}
SNOWFLAKE_USER={credentials['user']}
SNOWFLAKE_PASSWORD={credentials['password']}
SNOWFLAKE_WAREHOUSE={credentials['warehouse']}
SNOWFLAKE_ROLE={credentials['role']}
SNOWFLAKE_SCHEMA=PUBLIC

# User configuration
USER_NAME={credentials['user_name']}

# Azure Storage account
AZURE_STORAGE_ACCOUNT={credentials['azure_storage']}

# Schema evolution flag
EVOLVE_SCHEMA=True
"""
    
    env_file = temp_dir / ".env"
    env_file.write_text(env_content)
    print(f"Created temporary .env file at {env_file}")
    
    return env_file

def setup_project(temp_dir, credentials):
    """Set up the project structure"""
    print_step(3, "INITIALIZE PROJECT")
    
    # Path to the CLI script
    cli_script = Path(__file__).parent.parent / "cli.py"
    
    # Run the CLI scaffolder
    cmd = [
        "python3", str(cli_script), "init-project",
        "--name", credentials["user_name"],
        "--snowflake-account", credentials["account"],
        "--snowflake-user", credentials["user"], 
        "--snowflake-password", credentials["password"],
        "--azure-storage", credentials["azure_storage"],
        "--warehouse", credentials["warehouse"],
        "--role", credentials["role"],
        "--output-dir", str(temp_dir),
        "--evolve-schema"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print("Project initialized successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error initializing project: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def copy_sample_schemas(temp_dir):
    """Copy sample schemas to the temp directory"""
    print_step(4, "COPY SAMPLE SCHEMAS")
    
    # Source directory with sample schemas
    source_dir = Path(__file__).parent.parent / "local_schemas"
    target_dir = temp_dir / "local_schemas"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all sample schema files
    count = 0
    for file_path in source_dir.glob("*.sql.example"):
        # Create actual SQL file (without .example extension)
        target_path = target_dir / file_path.name.replace(".example", "")
        shutil.copy(file_path, target_path)
        print(f"Copied {file_path.name} to {target_path}")
        count += 1
    
    print(f"Copied {count} schema files to {target_dir}")

def run_etl(temp_dir):
    """Run the ETL process"""
    print_step(5, "RUN ETL PROCESS")
    
    # Add the temp directory to Python path
    sys.path.append(str(temp_dir))
    
    # Copy necessary modules to the temp directory
    source_dir = Path(__file__).parent.parent
    for module in ["rahil", "models", "migrations", "config"]:
        if (source_dir / module).exists():
            shutil.copytree(source_dir / module, temp_dir / module, dirs_exist_ok=True)
    
    # Copy additional files
    for file in ["alembic.ini", "cli.py"]:
        if (source_dir / file).exists():
            shutil.copy(source_dir / file, temp_dir / file)
    
    # Run the ETL process
    cli_script = temp_dir / "cli.py"
    cmd = ["python3", str(cli_script), "run", "--evolve"]
    
    try:
        print("Starting ETL process... (this may take a few minutes)")
        process = subprocess.Popen(
            cmd, 
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"ETL process failed with return code {return_code}")
        else:
            print("ETL process completed successfully!")
        
    except Exception as e:
        print(f"Error running ETL process: {e}")
        sys.exit(1)

def show_row_counts(temp_dir, credentials):
    """Show row counts for all tables"""
    print_step(6, "SHOW ROW COUNTS")
    
    # Import Snowflake connector
    try:
        import snowflake.connector
    except ImportError:
        print("Error: snowflake-connector-python is required.")
        print("Please install it with: pip install snowflake-connector-python")
        return
    
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=credentials['user'],
            password=credentials['password'],
            account=credentials['account'],
            warehouse=credentials['warehouse'],
            role=credentials['role']
        )
        
        # Create cursor
        cursor = conn.cursor()
        
        # Use the database
        database_name = f"IMT577_DW_{credentials['user_name']}_STAGING"
        cursor.execute(f"USE DATABASE {database_name}")
        cursor.execute("USE SCHEMA PUBLIC")
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print(f"\nDatabase: {database_name}")
        print(f"Tables: {len(tables)}")
        print("\nRow counts:")
        print("-" * 50)
        print(f"{'Table Name':<30} {'Row Count':>10}")
        print("-" * 50)
        
        # Get row count for each table
        for table_info in tables:
            table_name = table_info[1]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"{table_name:<30} {row_count:>10}")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("SNOWFLAKE ETL DEMO SCRIPT")
    print("=" * 80)
    print("\nThis script demonstrates the end-to-end ETL process.")
    print("It will create a temporary environment and run the ETL pipeline.")
    
    # Get credentials
    credentials = get_credentials_from_env()
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"\nCreated temporary directory: {temp_dir}")
        
        try:
            # Create environment file
            create_temp_env_file(credentials, temp_dir)
            
            # Set up project
            setup_project(temp_dir, credentials)
            
            # Copy sample schemas
            copy_sample_schemas(temp_dir)
            
            # Run the ETL process
            run_etl(temp_dir)
            
            # Show row counts
            show_row_counts(temp_dir, credentials)
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
        except Exception as e:
            print(f"\nError in demo script: {e}")
        finally:
            print("\n" + "=" * 80)
            print("DEMO COMPLETED")
            print("=" * 80)
            print(f"\nTemporary directory will be deleted: {temp_dir}")

if __name__ == "__main__":
    main() 