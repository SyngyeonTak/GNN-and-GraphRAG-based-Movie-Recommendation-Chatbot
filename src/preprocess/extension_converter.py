import pandas as pd
import os

def convert_ratings_to_csv(data_path: str, save_path: str):
    """
    Reads the ratings.dat file from MovieLens 1M, converts it to csv, and saves it.

    Args:
        data_path (str): The folder path containing the original .dat file.
        save_path (str): The folder path to save the converted .csv file.
    """
    print("Starting conversion of ratings.dat...")
    
    # Path to the .dat file
    dat_file = os.path.join(data_path, 'ratings.dat')
    
    # Define column names for the .dat file (based on the README)
    column_names = ['userId', 'movieId', 'rating', 'timestamp']
    
    # Read the .dat file using pandas' read_csv
    try:
        ratings_df = pd.read_csv(dat_file, 
                                 sep='::', 
                                 header=None, 
                                 names=column_names, 
                                 engine='python')
        
        # Save the processed data to a .csv file
        csv_file = os.path.join(save_path, 'ratings.csv')
        # index=False: Prevents writing the DataFrame index as a column
        ratings_df.to_csv(csv_file, index=False)
        
        print(f"✅ 'ratings.csv' has been successfully saved to the '{save_path}' folder.")
        print("Sample data:")
        print(ratings_df.head())
        
    except FileNotFoundError:
        print(f"[ERROR] '{dat_file}' not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred while processing ratings.dat: {e}")

def convert_movies_to_csv(data_path: str, save_path: str):
    """
    Reads the movies.dat file from MovieLens 1M, converts it to csv, and saves it.

    Args:
        data_path (str): The folder path containing the original .dat file.
        save_path (str): The folder path to save the converted .csv file.
    """
    print("\nStarting conversion of movies.dat...")

    # Path to the .dat file
    dat_file = os.path.join(data_path, 'movies.dat')
    
    # Define column names for the .dat file
    column_names = ['movieId', 'title', 'genres']
    
    # Read the .dat file using pandas
    try:
        movies_df = pd.read_csv(dat_file, 
                                sep='::', 
                                header=None, 
                                names=column_names, 
                                engine='python',
                                encoding='ISO-8859-1') # To prevent errors with special characters
        
        # Save the processed data to a .csv file
        csv_file = os.path.join(save_path, 'movies.csv')
        movies_df.to_csv(csv_file, index=False)
        
        print(f"✅ 'movies.csv' has been successfully saved to the '{save_path}' folder.")
        print("Sample data:")
        print(movies_df.head())

    except FileNotFoundError:
        print(f"[ERROR] '{dat_file}' not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred while processing movies.dat: {e}")

def main():
    """
    Main function to execute the data conversion tasks.
    """
    # Folder path containing the original MovieLens 1M data
    DATA_PATH = './dataset/raw/ml-1m/' 
    
    # Folder path to save the converted CSV files
    SAVE_PATH = './dataset/raw/ml-1m_v2/'

    # Create the save directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Function calls
    convert_ratings_to_csv(data_path=DATA_PATH, save_path=SAVE_PATH)
    convert_movies_to_csv(data_path=DATA_PATH, save_path=SAVE_PATH)
    
    print("\nAll .dat file conversions to CSV are complete.")


if __name__ == "__main__":
    # Call the main() function only when this script is executed directly
    main()