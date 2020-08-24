echo Starting installation
python -m venv bin/.venv
source bin/.venv/bin/activate
python -m pip install --upgrade pip
pip install -r bin/requirements.txt
echo Installation script finished