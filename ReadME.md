# Installation Guide
1. Clone this repo
2. Install [VSCode](https://code.visualstudio.com/download)
3. Install [Docker](https://docs.docker.com/get-started/get-docker/)
4. Install [Devcontainer](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension on VSCode
5. Open this repo with VSCode `code path_to_repo/Toy_Model_RL`
6. A pop up should appear that suggest you to rebuild and open the repo inside the devcontainer. Click it. If it doesn't appear you can trigger it manually by open the search bar with CTRL+SHIFT+P and searching "Rebuild and Reopen in Container"
7. Create two empty folders "logs" and "models"
8. You should be good to go! You can run the simulation using `pdm run python train.py`
