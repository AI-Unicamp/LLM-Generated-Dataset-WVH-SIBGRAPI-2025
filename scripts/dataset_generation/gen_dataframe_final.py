import pandas as pd

goemo_df_path = 'dataframes/52k_goemotions_filtered.csv'
tweet_df_path = 'dataframes/tweet_intensity_filtered.csv'
isear_df_path = 'dataframes/isear_dataset_filtered.csv'

df_final = pd.DataFrame()

df_final['Text'] = pd.concat([pd.read_csv(goemo_df_path)['text'], #.sample(n=3334, random_state=42),
                      pd.read_csv(tweet_df_path)['Text'], #.sample(n=3333, random_state=42),
                      pd.read_csv(isear_df_path)['txt']], #.sample(n=3333, random_state=42)],
                      ignore_index=True)

df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = 'dataframes/final_dataframe_all.csv'
df_final.to_csv(output_path, index=False)
