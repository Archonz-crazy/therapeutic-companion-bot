class csv:
    def convert_parquet_to_csv(directory):
        try:
            # Create a new directory for the CSV files
            csv_directory = os.path.join(directory, 'csv')
            os.makedirs(csv_directory, exist_ok=True)

            for filename in os.listdir(directory):
                if filename.endswith(".parquet"):
                    df = pd.read_parquet(os.path.join(directory, filename))
                    # Save the CSV files in the new directory
                    df.to_csv(os.path.join(csv_directory, filename[:-8] + '.csv'), index=False)
            print("All parquet files have been converted to CSV.")
        except Exception as e:
            print(f"Error: {e}")