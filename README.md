# Red_Light_Approach

Installation (OSX):

Install miniconda
Create conda env:\
```conda env create -f environment.yml```\
Install Homebrew:\
```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```\
Install XQuartz:\
```brew install --cask xquartz```
Use homebrew to install SUMO:\
```brew tap dlr-ts/sumo```\
```brew install sumo```\
Add SUMO_PATH environment variable in your ```.bashrc``` or ```.zshrc```:\
```export SUMO_PATH=/path/to/sumo```
Change paths in .sumocfg:\
```Do stuff so its not personal info```
Start XQuartz application from spotlight search or something
Run scenario with\
```python uniform_red_light_approach.py -h```
