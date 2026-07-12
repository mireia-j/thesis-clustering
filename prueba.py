import pandas as pd
import os

import pandas as pd
import os

def combine_csv_files(file_list, output_filename):
    if not file_list:
        print("No files provided in the list.")
        return

    dataframes = []
    first_df = None

    for file in file_list:
        if not os.path.exists(file):
            print(f"Warning: Could not find file -> {file}. Skipping...")
            continue

        # FIX: sep=None and engine='python' forces pandas to auto-detect commas OR semicolons
        df = pd.read_csv(file, sep=None, engine='python')
        
        # Clean up hidden whitespace in column names just in case
        df.columns = df.columns.str.strip()
        
        # Check if it's the first file to set the standard columns
        if first_df is None:
            first_df = df
            dataframes.append(df)
        else:
            # Check if columns match the first file
            if list(df.columns) != list(first_df.columns):
                print(f"Error: {file} has different columns than the first file.")
                print(f"First file columns ({len(first_df.columns)}): {list(first_df.columns)[:5]}...")
                print(f"This file columns ({len(df.columns)}): {list(df.columns)[:5]}...")
                return
            dataframes.append(df)

    if not dataframes:
        print("No valid CSV files were loaded. Nothing to save.")
        return

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save to a standard comma-separated CSV file
    combined_df.to_csv(output_filename, index=False)
    print(f"Successfully saved combined data to {output_filename}")


if __name__ == "__main__":
    folder = "Real_results_BEST_COMBO_filtered"
    
    file_list = [
        f"{folder}/euc_eqfcm.csv", 
        f"{folder}/mah_eqfcm.csv", 

    ]
    
    output_name = f"{folder}/FINAL_eqfcm.csv"

    combine_csv_files(file_list, output_name)

