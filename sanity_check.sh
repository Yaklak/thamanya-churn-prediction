#!/usr/bin/env bash
set -e

echo "🔍 Running sanity checks for churn-prediction project..."

# 1) Ensure base is deactivated
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
  echo "⚠️  Base environment active, deactivating..."
  conda deactivate
fi

# 2) Activate thamanya environment
echo "👉 Activating 'thamanya' env..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thamanya

# 3) Check Python version + path
echo "🐍 Python info:"
python -V
echo "PYTHONPATH: $PYTHONPATH"

# 4) Check that we are in the project root
echo "📂 Current directory: $(pwd)"

# 5) Verify __init__.py files
echo "🧩 Checking for __init__.py in src modules..."
missing_inits=false
for d in src src/data src/features src/models src/utils; do
  if [[ ! -f "$d/__init__.py" ]]; then
    echo "❌ Missing: $d/__init__.py"
    missing_inits=true
  else
    echo "✅ Found: $d/__init__.py"
  fi
done
if [ "$missing_inits" = true ]; then
  echo "⚠️ Please add missing __init__.py files."
fi

# 6) Try importing the module
echo "🧪 Testing import..."
python - <<'PY'
import sys, os
sys.path.append(os.getcwd())  # ensure project root in path
try:
    from src.data.load import load_dataset
    print("✅ Import succeeded: src.data.load.load_dataset available")
except Exception as e:
    print("❌ Import failed:", e)
PY

echo "✅ Sanity checks completed."