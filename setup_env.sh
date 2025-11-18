#!/bin/bash

# Automated setup script for gRNA classification project
# Usage: chmod +x setup_env.sh && ./setup_env.sh

set -e  # Exit on error

echo "=========================================="
echo "gRNA Classification - Environment Setup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements_ml.txt" ]; then
    echo "❌ Error: requirements_ml.txt not found"
    echo "Please run this script from the grna-inspector project root"
    exit 1
fi

echo "📦 Step 1: Creating virtual environment..."
python3 -m venv venv
echo -e "${GREEN}✓${NC} Virtual environment created"
echo ""

echo "🔧 Step 2: Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

echo "⬆️  Step 3: Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

echo "📚 Step 4: Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements_ml.txt --quiet
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

echo "📓 Step 5: Installing Jupyter and ipykernel..."
pip install jupyter ipykernel --quiet
echo -e "${GREEN}✓${NC} Jupyter installed"
echo ""

echo "🔬 Step 6: Creating Jupyter kernel..."
python -m ipykernel install --user --name=grna-env --display-name="Python (gRNA)" 2>/dev/null
echo -e "${GREEN}✓${NC} Jupyter kernel created"
echo ""

echo "🛠️  Step 7: Installing project in development mode..."
# Create __init__.py if it doesn't exist
mkdir -p src/grna_inspector
touch src/grna_inspector/__init__.py

# Create minimal setup.py if it doesn't exist
if [ ! -f "setup.py" ]; then
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="grna_inspector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
EOF
fi

pip install -e . --quiet
echo -e "${GREEN}✓${NC} Project installed"
echo ""

echo "✅ Verification..."
python -c "import numpy, pandas, sklearn, xgboost; print('  • Core packages OK')" 2>/dev/null || echo "  ⚠️  Warning: Some packages may not have installed correctly"
python -c "import sys; print(f'  • Python: {sys.version.split()[0]}')"
python -c "import sys; print(f'  • Path: {sys.executable}')"
echo ""

echo "=========================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "To start Jupyter Notebook:"
echo -e "  ${YELLOW}jupyter notebook${NC}"
echo ""
echo "To start Jupyter Lab:"
echo -e "  ${YELLOW}jupyter lab${NC}"
echo ""
echo "In Jupyter, select kernel:"
echo "  Kernel → Change kernel → Python (gRNA)"
echo ""
echo "To train baseline models:"
echo -e "  ${YELLOW}python train_baseline.py --data_dir data --output_dir models${NC}"
echo ""
echo "=========================================="
