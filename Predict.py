import sys
import pandas as pd
import requests
import base64
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib


def extract_metadata(file_path):
    try:
        if file_path.lower().endswith('.mp3'):
            audio = MP3(file_path, ID3=EasyID3)
        elif file_path.lower().endswith('.flac'):
            audio = FLAC(file_path)
        else:
            print("Unsupported file format")
            return None

        metadata = {
            'title': audio.get('title', [None])[0],
            'artist': audio.get('artist', [None])[0],
            'album': audio.get('album', [None])[0],
            'year': audio.get('date', [None])[0],
            'track_number': audio.get('tracknumber', [None])[0],
            'genre': audio.get('genre', [None])[0],
            'composer': audio.get('composer', [None])[0],
            'performer': audio.get('performer', [None])[0],
            'length': audio.info.length if hasattr(audio, 'info') else None,
            'bitrate': audio.info.bitrate if hasattr(audio, 'info') else None,
            'sample_rate': audio.info.sample_rate if hasattr(audio, 'info') else None,
            'channels': audio.info.channels if hasattr(audio, 'info') else None,
        }

        return metadata

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_song_features(song_file):
    metadata = extract_metadata(song_file)
    if metadata:
        df = pd.DataFrame([metadata])
    else:
        print("No metadata extracted.")

    song_name = df.loc[0, 'title']
    print('The song to predict is :', song_name)

    client_id = 'f18099689fdd43ac9ced954177fcd3e6'
    client_secret = 'd0930ca7a5aa4cf6a138a61d4660c165' # use your id and key
    # fill in and comment out assert 
    assert client_secret != 'fill your secret'
    # Encode client ID and client secret
    client_creds = f"{client_id}:{client_secret}"
    client_creds_b64 = base64.b64encode(client_creds.encode())

    token_url = "https://accounts.spotify.com/api/token"
    token_data = {
        "grant_type": "client_credentials"
    }
    token_headers = {
        "Authorization": f"Basic {client_creds_b64.decode()}"
    }

    r = requests.post(token_url, data=token_data, headers=token_headers)
    token_response_data = r.json()
    access_token = token_response_data['access_token']

    search_url = "https://api.spotify.com/v1/search"
    search_headers = {
        "Authorization": f"Bearer {access_token}"
    }
    search_params = {
        "q": song_name,
        "type": "track",
        "limit": 1
    }

    r = requests.get(search_url, headers=search_headers, params=search_params)
    search_results = r.json()

    track_id = search_results['tracks']['items'][0]['id']

    audio_features_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    audio_features_headers = {
        "Authorization": f"Bearer {access_token}"
    }

    r = requests.get(audio_features_url, headers=audio_features_headers)
    audio_features = r.json()

    song_features = pd.DataFrame([audio_features])

    return song_features
    
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.45)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.45)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.45)
        
        self.fc4 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

def knn_approach(song_file):
    song_features = get_song_features(song_file)
    data = pd.read_csv('result.csv')

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.compose import ColumnTransformer

    y = data['popularity_level']
    X = data.drop(columns=['popularity', 'popularity_level'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

    # smote
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define the preprocessing steps
    preprocessor = ColumnTransformer([
        ('minmax', MinMaxScaler(), [
            'tempo', 'duration_ms', 'loudness', 
            'energy', 'speechiness', 'danceability', 'liveness', 
            'instrumentalness', 'valence', 'acousticness'
        ]),
        ('categorical', OneHotEncoder(), ['key'])
    ], remainder='passthrough')

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline

    # Create a pipeline for KNN
    pipeline_knn = make_pipeline(preprocessor, KNeighborsClassifier(
        n_neighbors=2,
    ))

    # Fit the model on the resampled training data
    pipeline_knn.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred_knn = pipeline_knn.predict(X_test)

    # Evaluate the model
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    train_score_knn = pipeline_knn.score(X_train_resampled, y_train_resampled)
    test_score_knn = pipeline_knn.score(X_test, y_test)
    cr_knn = classification_report(y_test, y_pred_knn)

    print(f"Model: KNeighborsClassifier")
    print(f"Accuracy on Test Set for KNeighborsClassifier = {accuracy_knn:}")
    print(f"Train Score for KNeighborsClassifier = {train_score_knn:}")
    print(f"Test Score for KNeighborsClassifier = {test_score_knn:}\n")
    print(cr_knn)

    result = pipeline_knn.predict(song_features)

    if result == [0]:
        print('The given song has low chance of getting popular')
    elif result == [1]:
        print('The given song has a moderate chance of getting popular')
    elif result == [2]:
        print('The given song has high chance of getting popular')

def svc_approach(song_file):
    song_features = get_song_features(song_file)

    data = pd.read_csv('result.csv')

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.compose import ColumnTransformer

    y = data['popularity_level']
    X = data.drop(columns=['popularity', 'popularity_level'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

    # smote
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define the preprocessing steps
    preprocessor = ColumnTransformer([
        ('minmax', MinMaxScaler(), [
            'tempo', 'duration_ms', 'loudness', 
            'energy', 'speechiness', 'danceability', 'liveness', 
            'instrumentalness', 'valence', 'acousticness'
        ]),
        ('categorical', OneHotEncoder(), ['key'])
    ], remainder='passthrough')

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline

    # Create the pipeline for SVC
    pipeline_svc = make_pipeline(preprocessor, SVC( kernel = 'poly', C=2))

    # Fit the model
    pipeline_svc.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred_svc = pipeline_svc.predict(X_test)

    # Evaluate the model
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    train_score_svc = pipeline_svc.score(X_train_resampled, y_train_resampled)
    test_score_svc = pipeline_svc.score(X_test, y_test)
    cr_svc = classification_report(y_test, y_pred_svc)

    print(f"Model: SVC")
    print(f"Accuracy on Test Set for SVC = {accuracy_svc}")
    print(f"Train Score for SVC = {train_score_svc}")
    print(f"Test Score for SVC = {test_score_svc}\n")
    print(cr_svc)

    result = pipeline_svc.predict(song_features)

    if result == [0]:
        print('The given song has low chance of getting popular')
    elif result == [1]:
        print('The given song has a moderate chance of getting popular')
    elif result == [2]:
        print('The given song has high chance of getting popular')

def rf_approach(song_file):
    song_features = get_song_features(song_file)
    data = pd.read_csv('result.csv')

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.compose import ColumnTransformer

    y = data['popularity_level']
    X = data.drop(columns=['popularity', 'popularity_level'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_squared_error, r2_score

    preprocessor_rf = ColumnTransformer([
        ('minmax', StandardScaler(), [
            'tempo', 'duration_ms', 'loudness', 
            'energy', 'speechiness', 'danceability', 'liveness', 
            'instrumentalness', 'valence', 'acousticness'
        ]),
        ('categorical', OneHotEncoder(), ['key'])
    ], remainder='passthrough')

    from sklearn.pipeline import make_pipeline

    # Initialize and train the RandomForestRegressor
    model_rf = make_pipeline(
        preprocessor_rf,
        RandomForestClassifier(n_estimators=250, 
                               max_depth=50, 
                               min_samples_leaf=7,
                               max_features='sqrt',
                               random_state=42)
    )
    model_rf.fit(X_train, y_train)

    # Make predictions
    y_pred = model_rf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    score1 = model_rf.score(X_train, y_train)
    score2 = model_rf.score(X_test, y_test)

    print(f"Model train score: {score1}")
    print(f"Model test score: {score2}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    result = model_rf.predict(song_features)

    if result == [0]:
        print('The given song has low chance of getting popular')
    elif result == [1]:
        print('The given song has a moderate chance of getting popular')
    elif result == [2]:
        print('The given song has high chance of getting popular')

def nn_approach(song_file):
    # Check if MPS is available and set the device accordingly
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Using device: {device}')

    # Load the preprocessing pipeline
    pipeline = joblib.load('preprocessing_pipeline_fine_tuned.pkl')

    # Load the entire model
    model = torch.load('model_fine_tuned.pth')
    model.to(device)
    model.eval()

    # New song data
    song_features = get_song_features(song_file)

    # Transform new data using the same pipeline
    song_transformed = pipeline.transform(song_features)

    # Convert to tensor and move to device
    song_tensor = torch.tensor(song_transformed, dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(song_tensor)
        _, predicted_classes = torch.max(predictions, 1)

    # Convert tensor to list and print (.numpy not working since lib conflict)
    result = predicted_classes.cpu().tolist()
    if result == [0]:
        print('The given song has low chance of getting popular')
    elif result == [1]:
        print('The given song has a moderate chance of getting popular')
    elif result == [2]:
        print('The given song has high chance of getting popular')

def main(command, song_file):

    if command == '-knn':
        knn_approach(song_file)
    elif command == '-svc':
        svc_approach(song_file)
    elif command == '-rf':
        rf_approach(song_file)
    elif command == '-nn':
        nn_approach(song_file)
    else:
        print('\n\nThe command isn\'t correct! Try -knn, -svc, -rf, -nn')



if __name__=='__main__':
    if len(sys.argv) < 3:
        print("\nCommand or song file not given!\nUsage: python Predict.py <-knn, -svc, -rf, -nn> test_song.flac")
        sys.exit(1)

    command = sys.argv[1]
    song_file = sys.argv[2]

    main(command, song_file)