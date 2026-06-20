# Create a new venv
/opt/homebrew/bin/python3.12 -m venv ABM_venv

# Activate the venv
source ABM_venv/bin/activate

# Upgrade pip
ABM_venv/bin/pip install --upgrade pip

# Install packages
ABM_venv/bin/pip install -r requirements.txt

# Register the new kernel with Jupyter
ABM_venv/bin/pip install ipykernel
ABM_venv/bin/python -m ipykernel install --user --name ABM_venv --display-name "Python (ABM_venv)"

# Deactivate the venv
deactivate