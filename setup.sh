#!/usr/bin/env bash
# Setup script for STAGING_ETL environment

set -e

# 1. Clone the repository into /workspace
echo "Cloning STAGING_ETL repository..."
git clone https://github.com/<YOUR_GITHUB_USERNAME>/STAGING_ETL.git /workspace/STAGING_ETL

# 2. Enter project directory
cd /workspace/STAGING_ETL

# 3. Create and activate Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# 6. Initialize Alembic migrations (if not already initialized)
echo "Initializing Alembic migrations..."
if [ ! -d "migrations" ]; then
  alembic init migrations
fi

# 7. Copy example environment file for user configuration
echo "Copying example environment file..."
if [ -f rahil/example.env ]; then
  cp rahil/example.env rahil/.env
  echo ".env file created at rahil/.env. Please update it with your credentials."
else
  echo "Warning: rahil/example.env not found. Please create rahil/.env manually."
fi

# 8. Final message
echo "Setup complete. Environment is ready for offline use. Internet access will now be disabled." 