# Use an official Python runtime as a parent image
FROM mcr.microsoft.com/vscode/devcontainers/python:1-3.12

# Base ENV variables (anchor folders)
ENV HOME=/home/vscode

# Python-related env
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Pyenv
    PYENV_INSTALL_ROOT=${HOME}/.pyenv

# PATH (add utility scripts, pyenv bin, pdm bin, ruff)
ENV PATH=${PYENV_INSTALL_ROOT}/bin:${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}

RUN apt-get update && apt-get install -y \
    make curl

USER vscode

# Install pyenv (python versions manager see https://github.com/pyenv/pyenv)
RUN curl https://pyenv.run | PYENV_ROOT=${PYENV_INSTALL_ROOT} bash && \
    echo "# Pyenv" >> ~/.bashrc && \
    echo "eval \"\$(pyenv init -)\"" >> ~/.bashrc && \
    echo "# Pyenv" >> ~/.zshrc && \
    echo "eval \"\$(pyenv init -)\"" >> ~/.zshrc

# Install PDM (package and dependencies manager, see https://pdm.fming.dev)
RUN pipx install pdm==2.22.1 && \
    pdm config python.use_venv true && \
    pdm config venv.in_project true && \
    pdm config check_update false

# Set the working directory in the container
WORKDIR /workspaces/Toy_Model_RL

# Command to keep the container running (useful for development)
CMD ["sleep", "infinity"]