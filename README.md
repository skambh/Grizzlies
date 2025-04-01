source# Grizzlies
CSE 584 Final Project WN 25

---

Kiran Bodipati
Shruti Jain
Shashank Kambhammettu
Jai Narayanan

Usage

---

Clone repo: https://github.com/skambh/Grizzlies
Create virtual environment (venv or conda)
pip install -r requirements.txt
python3 -m pip install --upgrade build
python3 -m build
pip install Grizzlies
import grizzlies

After updating source:
python -m build
pip install dist/grizzlies-0.0.1-py3-none-any.whl --force-reinstall

Run test-temp-only using python -m tests.test-temp-only

![Tests](https://github.com/skambh/Grizzlies/actions/workflows/run_tests.yml/badge.svg)
