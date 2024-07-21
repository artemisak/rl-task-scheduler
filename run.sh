#!/bin/bash

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
else
    echo "Unsupported operating system. Exiting."
    exit 1
fi

echo "Detected operating system: $OS"

# Function to activate virtual environment
activate_venv() {
    if [ "$OS" == "Windows" ]; then
        source env/Scripts/activate
    else
        source env/bin/activate
    fi
}

# Function to check if all requirements are satisfied
check_requirements() {
    local missing_packages=0
    while IFS= read -r requirement || [[ -n "$requirement" ]]; do
        package=$(echo "$requirement" | cut -d'=' -f1)
        if ! pip list | grep -q "^$package "; then
            echo "Package not found: $package"
            missing_packages=$((missing_packages + 1))
        fi
    done < requirements.txt
    return $missing_packages
}

# Check if virtual environment exists
if [ ! -d "env" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment detected. Creating one..."
    if [ "$OS" == "Windows" ]; then
        python -m venv env
    else
        python3 -m venv env
    fi
    activate_venv
    echo "Installing requirements..."
    pip install -r requirements.txt
elif [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating existing virtual environment..."
    activate_venv
    if check_requirements; then
        echo "All requirements are already satisfied."
    else
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    fi
else
    echo "Already in a virtual environment."
    if check_requirements; then
        echo "All requirements are already satisfied."
    else
        echo "Installing missing requirements..."
        pip install -r requirements.txt
    fi
fi

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set default values for arguments
MAX_CAPACITY=3
MAX_AGENTS=21
MAX_DAYS=7
MAX_EPISODE_LENGTH=7
ALGORITHM="MADDPG"

# Run training script
echo "Starting training..."
python train.py --max_capacity $MAX_CAPACITY --max_agents $MAX_AGENTS --max_days $MAX_DAYS --max_episode_length $MAX_EPISODE_LENGTH --algorithm $ALGORITHM --timestamp $TIMESTAMP

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

# Run evaluation script
echo "Starting evaluation..."
python evaluate.py --max_capacity $MAX_CAPACITY --max_agents $MAX_AGENTS --max_days $MAX_DAYS --max_episode_length $MAX_EPISODE_LENGTH --algorithm $ALGORITHM --folder "${ALGORITHM}_${TIMESTAMP}"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Evaluation failed. Exiting."
    exit 1
fi

# Open TensorBoard
echo "Opening TensorBoard..."
tensorboard --logdir=./logs --port=6006 &

# Wait for TensorBoard to start
sleep 5

# Open TensorBoard in the default browser
if [ "$OS" == "macOS" ]; then
    open http://localhost:6006
elif [ "$OS" == "Linux" ]; then
    xdg-open http://localhost:6006
elif [ "$OS" == "Windows" ]; then
    start http://localhost:6006
else
    echo "Unable to automatically open browser. Please navigate to http://localhost:6006 to view TensorBoard."
fi

echo "Script completed. TensorBoard is running in the background."