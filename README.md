# red-light-approach
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

red-light-approach is a Python package that computes an optimal (w.r.t. time) motion plan during traffic light approach.



## Requirements

1. macOS (no instructions written for linux but it should be possible)

2. Miniconda or Anaconda\
To install, visit
[Conda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)

3. Homebrew\
To install, run:\
`$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`\
in your terminal window.



## Installation (macOS)

1. Install XQuartz with Homebrew:
   * `$ brew install --cask xquartz`

2. Install SUMO with Homebrew:
   * `$ brew tap dlr-ts/sumo`
   * `$ brew install sumo`

3. Edit `.bashrc` or `.zshrc`:
   * Add these lines to your shell's config file:
     ```
     export SUMO_HOME="/path/to/sumo"
     export PYTHONPATH="$PYTHONPATH:/path/to/parent/"
     ```
        * Replace `/path/to/sumo` with your sumo location, which you can find with `$ which sumo`.
        * Replace `/path/to/parent` with the directory into which you clone this repo, which you can find with `$ pwd`.

   * Load these environment variables with
     `$ source ~./bashrc` or `$source ~./zshrc`.

4. Clone this repository:
   * `$ git clone https://github.com/basilforlife/Red_Light_Approach.git`

5. Change directories to the root of Red_Light_Approach:
   * `$ cd Red_Light_Approach`

6. Create and activate conda env:
   * `$ conda env create -f environment.yml`
   * `$ conda activate RLA`

7. If you're going to contribute, add pre-commit hooks:
   * `$ pre-commit install`



## Usage

Confirm installation was successful with the test suite:
   * `$ pytest`

Start XQuartz from the application folder. This must be running in order to use the `-g` (graphical) option.

### Typical Use

Run the default scenario with the graphical simulation:
   * `$ python red_light_approach.py -c ./tests/test_params -g`

To speed up future runs use the the `-p <filename>` option:
   * `$ python red_light_approach.py -c ./tests/test_params -p test_params.pickle`

On subsequent runs, use `-u <filename>` to load the same configuration as before:
   * `$ python red_light_approach.py -u test_params.pickle`

To plot the result of N runs, use the `-N` option:
   * `$ python red_light_approach.py -u test_params.pickle -N 100`

