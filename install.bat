echo Starting installation

python -m venv bin\.venv
call bin\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r bin\requirements.txt

echo Installation script finished
pause