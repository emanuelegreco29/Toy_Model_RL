# RL Dogfighting Toy Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Toy Model** implementation to train Reinforcement Learning (RL) agents for **Within Visual Range (WVR) aerial combat** (dogfighting). This project serves as a sandbox environment to experiment with different RL algorithms, environment complexities, and training paradigms like self-play in a simplified 2D/3D simulation.

The goal is to provide a flexible and easy-to-use platform for researchers, students, and enthusiasts to explore the challenges of air combat maneuvering using modern RL techniques.

## ✨ Features

*   **Modular Environments**: Multiple environments with increasing complexity, from simple "tag" games to full dogfighting scenarios with shooting mechanics.
*   **Multiple Algorithms**: Implementations of standard (PPO) and advanced (IPPO) algorithms, with integrations for the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) library.
*   **Self-Play Support**: Scripts designed to train agents using Self-Play, a key technique for mastering competitive games.
*   **Dockerized Development**: A fully configured development container (Devcontainer) for a consistent and hassle-free setup across different machines.
*   **Evaluation Scripts**: Dedicated scripts to evaluate and visualize the performance of trained agents.
*   **Configurable Training**: Easily modify training parameters, reward functions, and environment settings.

## 🚀 Quick Start (Installation)

The project uses **VSCode Devcontainers** and **PDM** for a seamless setup. Follow these steps to get started:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/emanuelegreco29/RL_Dogfighting_Toy_Model.git
    cd RL_Dogfighting_Toy_Model
    ```

2.  **Install Prerequisites**
    *   Install [Visual Studio Code](https://code.visualstudio.com/)
    *   Install [Docker](https://www.docker.com/products/docker-desktop/)
    *   Install the **Dev Containers** extension in VSCode (`ms-vscode-remote.remote-containers`).

3.  **Open in Devcontainer**
    *   Open the cloned repository folder in VSCode: `code .`
    *   A pop-up should appear in the bottom right corner suggesting you "Reopen in Container". Click it.
    *   If the pop-up doesn't appear, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and run the command: **"Dev Containers: Rebuild and Reopen in Container"**.

4.  **Create Log Directories**
    *   Once the container is built and loaded, create two empty folders in the root of the project:
        ```bash
        mkdir logs models
        ```

5.  **Run a Test Training**
    *   You can now run a simple training script to verify everything works:
        ```bash
        pdm run python train/train.py
        ```

## 🧠 Project Structure

Here's a breakdown of the main directories and their purpose:

### `environments/`
Contains the core simulation logic. Each file defines a different `gym.Env` environment.

*   **`tag_env.py`**: A simple "tag" game where one agent tries to catch the other. Good for debugging basic RL logic.
*   **`shootdown_env.py`**: An environment where an agent must not only pursue but also "shoot" down a target.
*   **`fight_env.py`**: The core dogfighting environment, involving multiple agents in a competitive setup.
*   **`tag_shoot_env.py`**: A combined environment, merging the pursuit of 'tag' with the offensive action of 'shoot'.

### `algorithms/`
Houses the custom implementations of RL algorithms.

*   **`simple_ppo.py`**: A from-scratch implementation of Proximal Policy Optimization (PPO).
*   **`IPPO/`**: Implementation of Independent PPO, a multi-agent variant where each agent learns its own policy.

### `train/`
Scripts for launching training runs with various configurations.

*   **`train.py`**: A basic training script.
*   **`fight_train.py`**, **`tag_train.py`**, **`shootdown_train.py`**: Environment-specific training scripts.
*   **`tag_shoot_train.py`**: Standard training for the combined environment.
*   **`tag_shoot_ippo.py`**: Training script specifically using the IPPO algorithm.
*   **`tag_shoot_SAMC.py`**: Implements a Self-Play or advanced multi-agent curriculum.
*   **`dogfight_sb3.py`**: Training script using the Stable-Baselines3 library.

### `evaluate/`
Scripts to load trained models and visualize/evaluate their performance.

*   **`evaluate_model.py`**: A generic evaluation script.
*   **`evaluate_fight.py`**, **`evaluate_tag.py`**, etc.: Environment-specific evaluators.
*   **`evaluate_sb3_sp.py`**: Evaluation for models trained with Stable-Baselines3 and Self-Play.

### `SB3/`
Contains configurations, and trained models related to **Stable-Baselines3** experiments:

*   **`SB3_PPO_5_obs/`**, **`SB3_PPO_8_obs/`**: Experiments with different observation spaces.
*   **`SB3_PPO_Bezier/`**: Using Bézier curves for trajectory pursuit.
*   **`SB3_PPO_Moving/`**: Training against a moving target.
*   **`SB3_DQN_Moving/`**: Using DQN instead of PPO.
*   **`Fight/`**, **`UCAV/`**: Specific scenarios.

## 🎯 Training an Agent

You can train agents using either the custom algorithm implementations or the Stable-Baselines3 integration.

### Option 1: Using Custom Implementations

Navigate to the `train/` directory and run the appropriate script. For example, to train an agent in the `TagShoot` environment with IPPO:

```bash
pdm run python train/tag_shoot_ippo.py
```

You can modify hyperparameters directly within these training scripts.

### Option 2: Using Stable-Baselines3 (SB3)

For a wider range of tried-and-tested algorithms, you can use the SB3 scripts.

## 📊 Evaluating a Model

After training, you can evaluate a saved model using the scripts in the `evaluate/` directory.

```bash
# Example: Evaluate a trained model on the Tag environment
pdm run python evaluate/evaluate_tag.py --model-path /path/to/your/model.zip
```

These scripts typically render the environment, allowing you to visually inspect the agent's behavior and print episode statistics.

## 🛠️ Customization

*   **New Environments**: Create a new `.py` file in `environments/` following the Gymnasium API structure.
*   **Reward Functions**: Modify or create new reward functions in `reward_functions/` to shape agent behavior.
*   **Network Architectures**: Customize the policy networks `algorithms/`.
*   **Training Parameters**: Tweak hyperparameters (learning rate, batch size, etc.) directly in the training scripts.

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
