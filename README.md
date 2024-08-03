# CMPT353_ALLIN
CMPT353_ALLIN Project 2024 Summer

# Song Popularity Prediction
---
## Environment Setup
We have set up a requirement file, please navigate to your terminal, activate conda or other virtual environment if you like (unless you want to install dependencies in your OS environment which is not recommanded), run the following code.
`pip install -r requirements.txt`

---

## How to use

### Quick Starter
If you don't care about data analysis, just want to tryout the prediction models we made, simply run the following command.

`python3 Predict.py <-knn, -svc, -rf, -nn> <path_to_your_song_file>`

Or if you are marking this project(most likely), run the command below, **we have a test song in the repository** for quick start.

`python3 Predict.py <-knn, -svc, -rf, -nn> test_song.flac`

**Note you have option to choose different prediction models `-knn` for K-Nearest Neighbors Classifier, `-svc` for Support Vector Classifier, `-rf` for Random Forest Classifier, `-nn` for Neural Network.**

### Try it from Scratch
Since we kept all the original dataset in the **DataProcessing&Analysis** folder, *billboard_hot_stuff.csv* and *spotify_data.csv*. We can start our journey from cleaning data with original dataset.

**Working Flow:**

- Run *CleanData.ipynb*
- Run *DataAnalysis.ipynb*
- Run *plot.ipynb*

**Neural Network Run:**

- Run *audio_extractor.ipynb*
- Run *CleanData.ipynb*
- Run *nn_baseline_model.ipynb*
- Run *nn_fine_tuned_model.ipynb*
- Run *nn_predict_baseline.ipynb*
- Run *nn_predict_fine_tuned.ipynb*

---

## File explanation 

| File Path & Name|Description |
| ----------- | ----------- |
| Root -> Predict.py | Quick Starter File|
| Root -> test_song.flac | test song |
| Root -> requirements.txt | environment file |
| Root -> result.csv | file copy of *CleanData.ipynb* output |
| Root -> model_fine_tuned.pth | the optimal model we trained for running *Predict.py* |
| Root -> preprocessing_pipeline_fine_tuned.pkl | the optimal pipeline used for running *Predict.py* |
| DataProcessing&Analysis -> billboard_hot_stuff.csv | Original Dataset |
| DataProcessing&Analysis -> spotify_data.csv | Original Dataset |
| DataProcessing&Analysis -> result.csv | output of *CleanData.ipynb* |
| DataProcessing&Analysis -> extracted_feature_to_predict.csv | output of *audio_extractor.ipynb* |
| DataProcessing&Analysis -> Seraphine,Jasmine Clarke,Absofacto-Childhood Dreams.flac | test song |
| DataProcessing&Analysis -> audio_extractor.ipynb | feature extractor of song file |
| DataProcessing&Analysis -> CleanData.ipynb | clean data process |
| DataProcessing&Analysis -> DataAnalysis.ipynb | data analysis process |
| DataProcessing&Analysis -> plot.ipynb | data visualization |
| DataProcessing&Analysis -> nn_baseline_model.ipynb | trains baseline model |
| DataProcessing&Analysis -> nn_fine_tuned_model.ipynb | trains fine tune model |
| DataProcessing&Analysis -> nn_predict_baseline.ipynb | predict using baseline model |
| DataProcessing&Analysis -> nn_predict_fine_tuned.ipynb | predict using fine tune model |
| DataProcessing&Analysis -> model_baseline.pth | baseline model package |
| DataProcessing&Analysis -> model_fine_tuned.pth | fine tune model package |
| DataProcessing&Analysis -> preprocessing_pipeline_baseline.pkl | baseline model pipeline |
| DataProcessing&Analysis -> preprocessing_pipeline_fine_tuned.pkl | fine tune model pipeline |


---


## Conclusion

In conclusion, the CMPT353 project successfully identified key audio features influencing song popularity. However, several challenges were encountered, such as data variability, feature dependency on Spotify, and class imbalance. To enhance the project, the following steps are essential:

- Incorporating temporal features
- Integrating additional data sources
- Refining data cleaning and feature engineering
- Applying advanced resampling techniques
- Experimenting with different models and hyperparameter tuning
- Mitigating overfitting
- Improving API management
- Integrating real-time data and user feedback

These improvements can significantly enhance prediction accuracy and reliability, providing deeper insights into the factors contributing to musical success.
