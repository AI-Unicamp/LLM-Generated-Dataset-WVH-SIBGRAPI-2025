import pandas as pd

goemotions_path1 = 'raw_data/goemotions_1.csv'
goemotions_path2 = 'raw_data/goemotions_2.csv'
goemotions_path3 = 'raw_data/goemotions_3.csv'

'''
Selecionando exemplos com (anger, disgust, joy, fear, sadness, surprise, disappointment, disapproval)
e retirando exemplos neutros e não claros.
'''

goemotions_df_concat = pd.concat([pd.read_csv(goemotions_path1), 
                                  pd.read_csv(goemotions_path2), 
                                  pd.read_csv(goemotions_path3)], 
                                  ignore_index=True)

goemotions_df_final = goemotions_df_concat.loc[(goemotions_df_concat['example_very_unclear'] == False) & 
                                    (goemotions_df_concat['neutral'] == 0) &
                                    (goemotions_df_concat['text'].apply(lambda x: x.count(" ") > 5)) & # Garantir que a sentença não é muito curta
                                    ((goemotions_df_concat['anger'] == 1) |
                                     (goemotions_df_concat['disgust'] == 1) |
                                     (goemotions_df_concat['joy'] == 1) |
                                     (goemotions_df_concat['fear'] == 1) |
                                     (goemotions_df_concat['sadness'] == 1) |
                                     (goemotions_df_concat['surprise'] == 1) |
                                     (goemotions_df_concat['disappointment'] == 1) |
                                     (goemotions_df_concat['disapproval'] == 1)),
                                    'text']

output_path = 'dataframes/52k_goemotions.csv'
goemotions_df_final.to_csv(output_path, index=False)