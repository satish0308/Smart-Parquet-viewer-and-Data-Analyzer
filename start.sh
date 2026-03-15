#!/bin/bash
set -e

# ============================================
#  Parquet Explorer - Start Script
# ============================================

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="parquet-explorer"
PORT="${PORT:-8505}"

echo "============================================"
echo "  Parquet Explorer - Setup & Launch"
echo "============================================"
echo ""

# --- Check for conda ---
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed or not in PATH."
    echo "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# --- Initialize conda for script use ---
eval "$(conda shell.bash hook)"

# --- Create or update conda environment ---
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda environment '${ENV_NAME}' found."
    echo "[INFO] Updating environment..."
    conda env update -n "$ENV_NAME" -f "$APP_DIR/environment.yml" --prune -q
else
    echo "[INFO] Creating conda environment '${ENV_NAME}'..."
    conda env create -f "$APP_DIR/environment.yml" -q
fi

# --- Activate environment ---
echo "[INFO] Activating environment '${ENV_NAME}'..."
conda activate "$ENV_NAME"

# --- Verify installation ---
echo "[INFO] Python: $(python --version)"
echo "[INFO] Streamlit: $(streamlit --version)"

# --- Launch the app ---
echo ""
echo "============================================"
echo "  Launching Parquet Explorer on port ${PORT}"
echo "  URL: http://localhost:${PORT}"
echo "============================================"
echo ""

cd "$APP_DIR"
streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.fileWatcherType auto \
    --browser.gatherUsageStats false \
    --theme.base "light" \
    --theme.primaryColor "#1f77b4"
