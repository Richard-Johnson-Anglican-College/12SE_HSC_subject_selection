import pandas as pd

df = pd.read_csv(r"c:\Temp\11-12 SE\12SE\12SE_HSC_subject_selection\training_data.csv")

# Extract all target subjects from the dataframe
subjects = pd.concat([
    df['target_subject_1'], 
    df['target_subject_2'], 
    df['target_subject_3'], 
    df['target_subject_4'], 
    df['target_subject_5'], 
    df['target_subject_6'], 
    df['target_subject_7'].dropna()
]).unique()

print(f"Total unique subjects in CSV: {len(subjects)}")
print(sorted(subjects))
