# Model Training
![pylint](https://img.shields.io/badge/Pylint-NA-lightgrey?logo=python&logoColor=white)
![coverage](https://img.shields.io/badge/Coverage-97-yellow?logo=python&logoColor=white)
![test_score](https://img.shields.io/badge/ML_Test_Score-98.8-yellow?logo=pytest)

This repository contains the DVC-powered machine learning pipeline for training the sentiment analysis model used in our application. 

The stucture of this repository was inspired by the [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science/tree/master).  

All training code can be found under `src`. Note that the actual preprocessing logic of the training pipeline has ben factored out into the [lib-ml repository](https://github.com/remla25-team12/lib-ml). As such, `src/preprocess_data.py` uses methods from this library.

# Getting started
## Requirements
- Linux or macOS (recommended operating system)
- Python 3.9 or higher with pip

## Install and run
1. Clone this repository and navigate into the root folder:
   ```bash
   git clone https://github.com/remla25-team12/model-training.git
   cd model-training
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   > We HIGHLY recommend using a virtual environment, such as `venv` or `conda`, to avoid conflicts with your own Python environments.

3. Link Google Drive as remote storage location in DVC. For ease of use, we will instruct you to set up a Service Account instead of OAuth.
   1. In Google Cloud Console, create a [Service Account](https://cloud.google.com/iam/docs/service-accounts-create#creating). 
      1. In Step 1, after choosing a name and ID, **take note of the email address shown**, then press `Create and continue`\
      ![alt text](imgs/sa_email.png)
      2. In Step 2, select the `owner` role, then click `Continue`
      3. In Step 3, add your own Google Account email address to _both_ role fields, then click `Done`
   2. Click on the email address and go to the `Keys` tab. Click `Add Key > Create New Key > JSON >  Create`. 
   ![Add key for service account](imgs/sa_key.png)
   3. A `.json` file will be downloaded automatically. Rename it to `gdrive_sa_credentials.json` and store it in this repo's root folder.
   4. In Google Drive, create a folder you want to use as your remote storage. 
Share this folder with the email address associated with the Service Account created earlier. 
   6. Navitage into the folder and **take note of the folder ID**.
   ![Google Drive folder ID](imgs/gdrive_folder_id.png)
   7. Run the following commands to initialize the remote Google Drive storage in DVC:
      ```bash
      dvc remote add myremote gdrive://<FOLDER_ID> -f # Folder ID from previous step
      dvc remote default myremote
      dvc remote modify myremote gdrive_acknowledge_abuse true
      dvc remote modify myremote gdrive_use_service_account true
      dvc remote modify myremote --local gdrive_service_account_json_file_path gdrive_sa_credentials.json # The Service Account key downloaded earlier
      ```

## Training the model
Run the training pipeline with DVC:
```bash
dvc repro
```

Then push the trained model to the remote storage:
```bash
dvc push
```

The training pipeline generates two model files:
- `Classifier_Sentiment_Model.joblib`: Trained sentiment classification model.
- `c1_BoW_Sentiment_Model.pkl`: Fitted Bag-of-Words vectorizer.

These are stored remotely. To download the most recent models from the remote, run:
```bash
dvc pull
```

## Training configuration and parameters
TODO explain the config.yaml and what people may want to play with while training.
   
### Code Formatting and Style
1. Run Black (code formatter):
   ```bash
   black src/
   ```

2. Run isort (import sorter):
   ```bash
   isort src/
   ```

3. Run Flake8 (style guide enforcement):
   ```bash
   flake8 src/
   ```

4. Run PyLint with custom rules (including nan-check):
```bash
pylint --clear-cache-post-run=y src tests pylint_nan_check setup.py
```
## Testing
### Basic Testing
Run the tests using pytest:
```bash
pytest tests/
```

### Full Test Suite with Coverage and Metrics
Run the complete test suite with coverage and ML test adequacy metrics:
```bash
python tests/run_tests.py
```

This command will:
- Run all tests
- Generate coverage reports
- Calculate ML test adequacy scores

## Linting
### PyLint
Project specific:
1. Naming: Add the relevant good names, including use of X, X_train, X_test etc, as they are informative in ML training.

## Use of Generative AI:
GitHub Copilot was used to understand and write the pytest cases for model-training.