# Hybrid Energy Forecasting and Trading Competition

This repository contains project implementation for the Hybrid Energy Forecasting and Trading Competition. Full details of the competition can be found on the [IEEE DataPort](https://dx.doi.org/10.21227/5hn0-8091).

This repository is based on [Getting started](https://github.com/jbrowell/HEFTcom24) examples by the organisers. 

## Prepare your python environment

Run
```
make virtualenv
source ./venv/bin/activate
```
to install all packages and dependencies, and activate the `HEFTcom24` environment.

## Download data

Historic data for use in the competition can be downloaded from the competition [IEEE DataPort page](https://dx.doi.org/10.21227/5hn0-8091) and should un-zipped and placed in the `data` folder.

More recent data can be downloaded via the competition API, which will be necessary to generate forecasts during the evaluation phase of the competition. Teams are advised to test this functionality early and automate submissions.

API documentation can be found on the rebase.energy website for [energy data](https://api.rebase.energy/challenges/redoc#tag/Data) and [weather data](https://api.rebase.energy/weather/docs/v2/), and some basic wrappers are included in `src/utils.py`.


## Utilities module

The python module `src` contains the functions for the competition participation, including

1. Set-up of authentication to access the competition API
2. Wrappers for the API endpoints to download the latest data
3. Functions to prepare and send submissions 

### Setting up your authentication

To use Rebase API, your API key should be stored in a text file called `team_key.txt` in the root directory of this repository. This file is listed in `.gitignore` and is therefore ignored by git. If you change the filename, make sure you add the new name to `.gitignore`.

## Submissions

During the competition period, daily submissions are required. Forecasts and market bids for the day-ahead must be submitted before gate closure of the day-ahead auction at 9:20AM UTC. Hence, automation is encouraged. Submission is via push API, documentation of which is available on the [rebase.energy website](https://api.rebase.energy/challenges/redoc#tag/Challenge/operation/post_submission_challenges__challenge_id__submit_post).

The python script `auto_submitter.py` provides an example of one way of setting this up. The script downloads new data, loads and runs models, and submits the resulting forecasts and market bid to the competition platform. It can be run from the command line by selecting among available pipelines (`gluonts`, `lgbm`, `baseline`)  

```sh
python src/auto_submitter.py --pipeline=lgbm 
```

## Automation

The automation of the submission is done with `prefect`. A [flow](https://docs.prefect.io/concepts/flows/) is the basis of all Prefect workflows.

```sh
python src --pipeline=lgbm
```

To view your flow runs from a UI, spin up a Prefect Orion server on your local machine:

```bash
prefect server start
```

Open the URL http://127.0.0.1:4200/ to see the Prefect UI.