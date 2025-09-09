import pandas as pd

isear_path = 'raw_data/isear_dataset.csv'

isear_df = pd.read_csv(isear_path, sep = ';', encoding='latin-1')

isear_df['txt'] = isear_df['txt'].str.replace('á\n', '', regex=False)

isear_df_final = isear_df.loc[
    (isear_df['txt'] != 'Not applicable to myself.') &
    (isear_df['txt'] != 'NO RESPONSE') &
    (isear_df['txt'] != '[ No response.]') &
    (isear_df['txt'].apply(lambda x: x.count(" ") > 5)), # Garantir que a sentença não é muito curta
    'txt'
]

output_path = 'dataframes/isear_dataset_2.csv'
isear_df_final.to_csv(output_path, index=False)


#(goemotions_df_concat['text'].apply(lambda x: x.count(" ") > 5)) & # Garantir que a sentença não é muito curta