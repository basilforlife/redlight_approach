[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# redlight_approach


redlight_approach is a Python package that computes an optimal (w.r.t. time) motion plan during traffic light approach.


https://user-images.githubusercontent.com/44418392/128266088-1e6e80fc-a898-4835-b7ac-15c516f53195.mov


## Requirements

1. Linux or macOS

2. Miniconda or Anaconda\
To install, visit
[Conda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)

3. SUMO: Installation instructions for [Linux (Ubuntu)](#install-sumo-ubuntu) and [macOS](#install-sumo-macos)


## Install SUMO (Ubuntu)

1. Build SUMO from source (see [SUMO Linux Build](https://sumo.dlr.de/docs/Installing/Linux_Build.html) for more details)
```
sudo apt-get install git cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev
git clone --recursive https://github.com/eclipse/sumo
export SUMO_HOME="$PWD/sumo"
mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)
sudo make install
```


## Install SUMO (macOS)

1. Install Homebrew if you don't have it:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Install XQuartz with Homebrew:
```
brew install --cask xquartz
```

3. Install SUMO with Homebrew:
```
brew tap dlr-ts/sumo
brew install sumo
```


## Installation of redlight_approach

3. Edit `.bashrc` or `.zshrc`:
  * Add these lines to your shell's config file:
    ```
    # Your .bashrc or .zshrc file
    
    export SUMO_HOME="/path/to/sumo"
    export PYTHONPATH="$PYTHONPATH:/path/to/parent/"
    ```
  * Replace `/path/to/sumo` above with your sumo location, which you can find with `which sumo`.
  * Replace `/path/to/parent` above with the directory into which you clone this repo, which you can find with `pwd`.

  * Load these environment variables with
    ```
    source ~/.bashrc
    # or 
    source ~/.zshrc
    ```

4. Clone this repository:
```
git clone https://github.com/basilforlife/redlight_approach.git
```

6. Change directories to the root of redlight_approach:
```
cd redlight_approach
```

7. Create and activate conda env:
```
conda env create -f environment.yml
conda activate rla
```

8. If you're going to contribute, add pre-commit hooks:
```
pre-commit install
```



## Usage

Confirm installation was successful with the test suite:
```
pytest
```

macOS: Start XQuartz from the application folder. This must be running in order to use the `-g` (graphical) option.

### Typical Use

Run the default scenario with the graphical simulation:
```
python simple_comparison.py -c parameter_files/original.json -g
```

To speed up future runs use the the `-p <filename>` option:
```
python simple_comparison.py -c parameter_files/original.json -p original.pickle
```

On subsequent runs, use `-u <filename>` to load the same configuration as before:
```
python simple_comparison.py -u original.pickle
```

To plot the result of N runs, use the `-N` option:
```
python simple_comparison.py -u original.pickle -N 100
```


## License

This work is licensed under the GNU Affero General Public License v3.0. Feel free to contact
me if you have any questions about the project.

