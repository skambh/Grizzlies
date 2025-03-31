source# Grizzlies
CSE 584 Final Project WN 25
_____________________
Kiran Bodipati
Shruti Jain
Shashank Kambhammettu
Jai Narayanan

Usage
_____
Clone repo: https://github.com/skambh/Grizzlies
Create virtual environment (venv or conda)
pip install -r requirements.txt
pip install -e . # for installing grizzlies

python -m tests.test-temp-only to run tests
After updating source:
python -m build
pip install dist/grizzlies-0.0.1-py3-none-any.whl --force-reinstall

Run test-temp-only using python -m tests.test-temp-only
