*"Devoted to those, who never fails to type 'predction' incorrectly"*

# A simple rainfall prediction repo
Some `yml` and `python` scripts to learn/practice ML Ops. Nothing fancy, just to have an ongoing MLOps project. The structure is straighforward:

## Workflow actions (YML files in `.github/workflows`)
As the name suggests these files impose scheduled/manually triggered actions. The actions invoke  python scripts on schedule or by manual trigger - for now:

### 1. Retrieve historical data from KNMI (`get_historical_rain_data.yml`, scheduled on Sundays at 13:00): 
**`set_dates_for_training.py`**

In order to get the historical dataset and make the `POST` request to KNMI website, we set the starting and end dates. Starting date is 10 years ago and end date is 1 month ago from today. This is manual for now but can and will be scheduled (like daily predcition) to be invoked in every month. 

**`get_historical_rain_data_from_knmi.py`**

Here, we make a `POST` request to KNMI's website (Dutch Meteoology website) using `requests` lib from python (quite handy, kudos to them). The retrieved file includes the actual rainfall data for the last 10 years for Eindhoven airport.

After getting the data, there is some data preparation going on to have a ready dataset for training the LGBM models ( feel free to call it *data munging, wrangling* etc for increased marketability).   

- **Input**: The `env` variables `starting date` and `ending date`
- **Output**: Historical rainfall amounts (`historical_rain_data.csv`)
 

### 2. Train the prediction models (`train_rainfall_model.yml`, scheduled on Sundays at 14:00)
**`train_rainfall_model.py`**

Here, we train two `LGBM` models, one regression and one classification. Regression model predicts the rainfall amount in mm. Classification predcits whether or now it will rain, i.e. rainfall>=0.1 mm. The two models are exclusive. 

Using classification models, we also provide the probability of rainfall. 

Here, obviously we are not trying to have a full blown weather model. The whole purpose is to have **A** decent model, and make reasonable predictions on schedule. 

- **Input**: Historical rainfall (`historical_rain_data.csv`)
- **Output**:
  - LGBM models - classification and regression (`rainfall_models.pickle`).
  - `train_test_set_comprehensive.csv` includes detailed data on training and testing set, errors, etc for tracking purposes if/when things go south.
  - `images` folder includes the confusion matrices. For example:
    
<img src="images/confusion_matrix_train.png" width="425"/> <img src="images/confusion_matrix_test.png" width="425"/> 

### 3.  Get daily data and make predictions for tomorrow (`predict_daily_rainfall.yml`, scheduled every day at 15:00 UTC)

**`predict_daily_rainfall.py`**

Using the saved models as `pickle` files, we get daily data similar to `get_daily_rain_data_from_knmi.py` and make predictions. Call it the inference part if you like.

- **Input**: LGBM models - classification and regression (`rainfall_models.pickle`)
- **Output**: Daily predictions file (`daily_prediction.csv`). Each row will be a prediction made at each day. For that reason, the file is written in `append` mode.
