import pandas as pd

def main():
    spotify_df = pd.read_csv('spotify_data.csv')
    billboard_df = pd.read_csv('billboard_hot_stuff.csv')

    # Standardize song names and artist names
    spotify_df['name'] = spotify_df['name'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    spotify_df['artists'] = spotify_df['artists'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    billboard_df['Song'] = billboard_df['Song'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    billboard_df['Performer'] = billboard_df['Performer'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)

    # Clean spotify dataset
    # Drop unnecessary columns
    columns_to_drop = ['explicit', 'id', 'mode', 'release_date']
    spotify_df = spotify_df.loc[:, ~spotify_df.columns.isin(columns_to_drop)]
    spotify_df = spotify_df.dropna()
    cleaned_spotify = spotify_df.drop_duplicates()

    # Clean billboard dataset
    # Select the required columns
    selected_columns = ['Song', 'Performer', 'Week Position', 'Weeks on Chart', 'Previous Week Position']
    df_selected = billboard_df[selected_columns].copy()
    # Ensure the 'Previous Week Position' is numeric
    df_selected['Previous Week Position'] = pd.to_numeric(df_selected['Previous Week Position'], errors='coerce')
    # Drop rows with NaN values in the 'Previous Week Position' column after conversion
    df_selected = df_selected.dropna(subset=['Previous Week Position'])
    # Calculate the maximum 'Weeks on Chart' for each song and performer
    df_max_weeks = df_selected.groupby(['Song', 'Performer'], as_index=False)['Weeks on Chart'].max()
    # Calculate the average 'Previous Week Position' for each song and performer
    df_avg_previous_week = df_selected.groupby(['Song', 'Performer'], as_index=False)['Previous Week Position'].mean()
    df_avg_previous_week.rename(columns={'Previous Week Position': 'Average Previous Week Position'}, inplace=True)
    # Convert the average 'Previous Week Position' to integer
    df_avg_previous_week['Average Previous Week Position'] = df_avg_previous_week['Average Previous Week Position'].round().astype(int)
    # Calculate the maximum 'Week Position' for each song and performer
    df_max_week_position = df_selected.groupby(['Song', 'Performer'], as_index=False)['Week Position'].min()
    # Merge the results
    cleaned_billboard = df_max_weeks.merge(df_avg_previous_week, on=['Song', 'Performer'])
    cleaned_billboard = cleaned_billboard.merge(df_max_week_position, on=['Song', 'Performer'])
    cleaned_billboard['Average Previous Week Position'] = round((100 - cleaned_billboard['Average Previous Week Position']) / 99 * 100)
    cleaned_billboard['Week Position'] = round((100 - cleaned_billboard['Week Position'])/ 99 * 100)

    # Create a combined key for better matching in cleaned spotify and billborad dataset
    cleaned_spotify = cleaned_spotify.copy() 
    cleaned_spotify.loc[:, 'combined_key'] = (cleaned_spotify['name'] + ' ' + cleaned_spotify['artists']).astype(str)
    cleaned_billboard = cleaned_billboard.copy()
    cleaned_billboard.loc[:, 'combined_key'] = (cleaned_billboard['Song'] + ' ' + cleaned_billboard['Performer']).astype(str)

    # Merge datasets on the key
    combined_df = pd.merge(cleaned_spotify, cleaned_billboard, on='combined_key', how='inner')
    # Drop duplicate rows in the combined dataframe
    combined_df = combined_df.drop_duplicates()

    # Drop unnecessary columns in the combined dataframe
    columns_to_drop_combined = ['Performer', 'name', 'combined_key']
    combined_df = combined_df.drop(columns=columns_to_drop_combined)

    def calculate_popularity(row):
        return (1 / (row['Week Position'] + 1)) * (1/3) + (1 / (row['Average Previous Week Position'] + 1)) * (1/3) + (row['Weeks on Chart'] / combined_df['Weeks on Chart'].max()) * (1/3)

    combined_df['new_popularity'] = combined_df.apply(calculate_popularity, axis=1)
    combined_df['new_popularity'] = combined_df['new_popularity'] * 100 / combined_df['new_popularity'].max()
    combined_df.loc[combined_df['popularity'] == 0, 'popularity'] = combined_df['new_popularity']

    columns_to_drop = ['Week Position', 'Average Previous Week Position', 'Weeks on Chart', 'new_popularity']
    combined_df.drop(columns=columns_to_drop, inplace=True)
    combined_df = combined_df.loc[combined_df.groupby('Song')['popularity'].idxmax()]

    combined_df.to_csv("result.csv", index = False)

if __name__=='__main__':
    main()