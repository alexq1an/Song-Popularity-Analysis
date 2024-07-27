import pandas as pd
import numpy as np
import seaborn as sns

def main():
    df = pd.read_csv('result.csv')
    df["duration_mins"] = df["duration_ms"]/60000
    df.drop(columns="duration_ms", inplace=True)
    df = df.sort_values(by='year', ascending=True)

    # drop unncecssary column
    df = df.drop(columns=['artists', 'Song'])

    correlation_matrix = df.corr()
    print(correlation_matrix)
    popularity_corr = correlation_matrix['popularity'].sort_values(ascending=False)
    print("Correlation with Popularity:\n", popularity_corr)

    data = df.copy()
    data.loc[((df.popularity >= 0) & (df.popularity <= 50)), "popularity_level" ] = 1
    data.loc[((df.popularity > 50) & (df.popularity <= 70)), "popularity_level" ] = 2
    data.loc[((df.popularity > 70) & (df.popularity <= 100)), "popularity_level" ] = 3
    data["popularity_level"] = data["popularity_level"].astype("int")
    data['popularity_level'].value_counts()
    
    from sklearn.model_selection import train_test_split
    # Define target variable 'y' and features 'X'
    y = data['popularity_level']
    X = data.drop(columns=['popularity_level'])
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    # Define the preprocessing steps
    preprocessor = ColumnTransformer([
        ('minmax', MinMaxScaler(), [
            'year', 'tempo', 'duration_mins', 'loudness', 
            'energy', 'speechiness', 'danceability', 'liveness', 
            'instrumentalness', 'valence',  
            'acousticness'
        ]),
        ('categorical', OneHotEncoder(), ['key'])
    ], remainder='passthrough')

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the training and test data
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    print(f"X_train_preprocessed shape: {X_train_preprocessed.shape}")
    print(f"X_test_preprocessed shape: {X_test_preprocessed.shape}")

    print(X_train_preprocessed.shape)
    print(y_train.shape)
    print(X_test_preprocessed.shape)
    print(y_test.shape)

    # knn
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

    results = []
    def run_model(model, alg_name, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cross_val_scores = cross_val_score(model, X_train, y_train, cv=6)
        cr = classification_report(y_test, y_pred)
        results.append((alg_name, accuracy, model))  
        # Print results
        print(f"Model: {alg_name}")
        print(f"Accuracy on Test Set for {alg_name} = {accuracy:.2f}\n")
        print(cr)
        print(f"{alg_name}: CrossVal Accuracy Mean: {cross_val_scores.mean():.2f} and Standard Deviation: {cross_val_scores.std():.2f} \n")
        
    # K-Nearest Neighbors Classifier
    model_knn = KNeighborsClassifier()
    run_model(model_knn, "Nearest Neighbors Classifier", X_train, X_test, y_train, y_test)

    # svc
    # Ensure stratified splitting of the data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Define and run the model
    def run_model(model, alg_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Perform Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=6)
        scores = cross_val_score(model, X_train, y_train, cv=skf)
        cr = classification_report(y_test, y_pred)
        # Store results
        results.append((alg_name, accuracy, model))
        print(f"Model: {alg_name}")
        print(f"Accuracy on Test Set for {alg_name} = {accuracy:.2f}\n")
        print(cr)
        print(f"{alg_name}: CrossVal Accuracy Mean: {scores.mean():.2f} and Standard Deviation: {scores.std():.2f} \n")

    # Example with SVC Classifier
    model_svc = SVC(kernel='poly', degree=3, C=1)
    run_model(model_svc, "SVC Classifier")

    # randomforest
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the training and test data
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    print(f"X_train_preprocessed shape: {X_train_preprocessed.shape}")
    print(f"X_test_preprocessed shape: {X_test_preprocessed.shape}")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    # Initialize and train the RandomForestRegressor
    model_rf = RandomForestRegressor(random_state=42)

    # use GridSearchCV to adjust parameters
    grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid,
                            cv=3, n_jobs=-1, verbose=2)

    grid_search.fit(X_train_preprocessed, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test_preprocessed)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    score1 = best_model.score(X_train_preprocessed, y_train)
    score2 = best_model.score(X_test_preprocessed, y_test)

    print(f"Model train score: {score1}")
    print(f"Model test score: {score2}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


    # kmean
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # Standardize the features
    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters (n_clusters) as needed
    kmeans.fit(X_scaled)

    # Predict the clusters
    data['cluster'] = kmeans.predict(X_scaled)

    # Visualize the clusters (optional, using PCA for 2D visualization)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    data['pca1'] = principal_components[:, 0]
    data['pca2'] = principal_components[:, 1]

    # Calculate inertia
    inertia = kmeans.inertia_
    print(f'Inertia: {inertia}')

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, data['cluster'])
    print(f'Silhouette Score: {silhouette_avg}')

if __name__=='__main__':
    main()