!#/bin/bash

python -m venv venv
source venv/Scripts/activate
pip install --no-deps pipreqs
pip install yarg==0.1.9 docopt==0.6.2
# python.exe -m pip install --upgrade pip
pip install -r requirements.txt
