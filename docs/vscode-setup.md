# VS Code Setup

Open `~/quant-lab` in VS Code and install the recommended extensions when
prompted.

Recommended workflow:

1. Run the task `Quant Lab: Sync Environment`.
2. Select `${workspaceFolder}/.venv/bin/python` as the Python interpreter.
3. Run the task `Quant Lab: Check`.
4. Start `Quant Lab: MLflow UI` when comparing experiments.

Use the Dev Containers extension when Docker is available. The dev container
installs Python, `uv`, Quarto, Git LFS, and the recommended VS Code extensions.
