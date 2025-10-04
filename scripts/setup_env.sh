#!/bin/bash
#
# Non-interactive script to install Miniconda and set up the vLLM environment.
#
# This script is designed for automation (CI/CD, server setup) and uses batch
# flags to avoid interactive prompts.

# --- Configuration ---
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
# Define installation path. Using $HOME/miniconda is standard.
INSTALL_PATH="$HOME/miniconda"
CONDA_ENV_NAME="hfax"
PYTHON_VERSION="3.12"

# Exit immediately if a command returns a non-zero status.
# Conda activation uses 'set +e' internally and can cause issues if not handled carefully, 
# but we will rely on the checks below.

echo "--- 1. Downloading Miniconda Installer ---"
# Check if the installer is already downloaded
if [ -f "$MINICONDA_INSTALLER" ]; then
    echo "Installer is already present: $MINICONDA_INSTALLER"
else
    # Download the latest Linux installer (using -O to name the file)
    wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O "$MINICONDA_INSTALLER"
    echo "Download complete."
fi

echo "--- 2. Installing Miniconda to $INSTALL_PATH (Non-Interactive) ---"
# Check if the installation path already exists to prevent the installer error.
if [ -d "$INSTALL_PATH" ]; then
    echo "Miniconda installation directory already exists at $INSTALL_PATH. Skipping installation."
else
    # -b: batch mode (non-interactive)
    # -p: specify installation prefix/path
    bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_PATH"
    echo "Installation finished."
fi

echo "--- 3. Initializing Conda for current script session ---"
# Source the initialization script to make the 'conda' command available for the script.
if [ -f "$INSTALL_PATH/etc/profile.d/conda.sh" ]; then
    source "$INSTALL_PATH/etc/profile.d/conda.sh"
    echo "Conda commands enabled for this script session."
else
    echo "Error: Conda initialization script not found at expected path."
    exit 1
fi

echo "--- 3.2. Configuring Conda for future interactive sessions (via 'conda init') ---"
# Configure the shell startup files (like ~/.bashrc) for permanent 'conda' command access.
# If this was already run, it will typically print a harmless warning/message.
conda init bash

echo "--- 3.3. Sourcing ~/.bashrc to load persistent configuration ---"
# Source the bash profile to activate the changes made by 'conda init', 
# typically enabling the (base) environment.
source ~/.bashrc
echo "Conda base environment configuration loaded from ~/.bashrc."

# --- 3.5. Accepting Conda Terms of Service (TOS) for Non-Interactive Use ---
echo "--- 3.5. Accepting Conda TOS for required channels ---"
# This is required for non-interactive execution of 'conda create' 
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo "Conda TOS accepted."

# --- 4. Create and Setup vLLM Conda Environment ---
echo "--- 4. Creating Conda environment: $CONDA_ENV_NAME (Python $PYTHON_VERSION) ---"

# Check if the environment already exists before attempting to create it
if conda info --envs | grep -q "^\s*$CONDA_ENV_NAME\s"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Skipping creation."
else
    # Create the environment non-interactively (-y)
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    echo "Environment '$CONDA_ENV_NAME' created successfully."
fi


echo "--- 5. Activating and Verifying Environment ---"
# Activate the environment in the current script shell
conda activate "$CONDA_ENV_NAME"
echo "Successfully activated environment: $CONDA_ENV_NAME"

# Verify the Python version inside the new environment
echo "Python version inside $CONDA_ENV_NAME:"
python --version

echo "--- Miniconda and vLLM Conda Setup Complete ---"
echo ""
echo "NOTE: The environment is active for the remainder of this script's execution."
echo "To use it in a new terminal session, the persistent setup is now complete."

# You can now add your "Build wheel from source" commands here using standard
# 'pip install' or 'python setup.py install' commands, which will execute inside the activated environment.
