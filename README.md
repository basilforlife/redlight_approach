# Red_Light_Approach
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Installation (OSX):

Install miniconda\
Create conda env:\
```conda env create -f environment.yml```\
Install Homebrew:\
```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```\
Install XQuartz:\
```brew install --cask xquartz```
Use homebrew to install SUMO:\
```brew tap dlr-ts/sumo```\
```brew install sumo```\
Add SUMO_HOME environment variable in your ```.bashrc``` or ```.zshrc```:\
```export SUMO_HOME=/path/to/sumo```\
Also add to this file the parent dir of project to PYTHONPATH:\
```export PYTHONPATH:$PYTHONPATH:/path/to/parent/```\
Change paths in .sumocfg:\
```Do stuff so its not personal info```\
Confirm things are working with test suite\
```pytest```\
Start XQuartz application from spotlight search or something\
Run scenario with\
```python red_light_approach.py -h```
