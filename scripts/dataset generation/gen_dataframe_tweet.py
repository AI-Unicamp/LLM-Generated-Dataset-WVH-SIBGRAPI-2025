import pandas as pd

dataset_path = 'raw_data/tweet_intensity_dataset/'

df_anger_train = pd.read_csv(dataset_path + 'anger_dataset_train.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_fear_train = pd.read_csv(dataset_path + 'fear_dataset_train.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_joy_train = pd.read_csv(dataset_path + 'joy_dataset_train.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_sadness_train = pd.read_csv(dataset_path + 'sadness_dataset_train.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])


df_anger_dev = pd.read_csv(dataset_path + 'anger_dataset_dev.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_fear_dev = pd.read_csv(dataset_path + 'fear_dataset_dev.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_joy_dev = pd.read_csv(dataset_path + 'joy_dataset_dev.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_sadness_dev = pd.read_csv(dataset_path + 'sadness_dataset_dev.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])

df_anger_test = pd.read_csv(dataset_path + 'anger_dataset_test.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_fear_test = pd.read_csv(dataset_path + 'fear_dataset_test.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_joy_test = pd.read_csv(dataset_path + 'joy_dataset_test.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])
df_sadness_test = pd.read_csv(dataset_path + 'sadness_dataset_test.csv', sep = '	', header = None, names=["ID", "Text", "Emotion", "Intensity"])

tweet_df_concat = pd.concat([df_anger_train,
                     df_fear_train,
                     df_joy_train,
                     df_sadness_train,
                     df_anger_dev,
                     df_fear_dev,
                     df_joy_dev,
                     df_sadness_dev,
                     df_anger_test,
                     df_fear_test,
                     df_joy_test,
                     df_sadness_test], 
                     axis=0, ignore_index=True)

tweet_df_final = tweet_df_concat.loc[(tweet_df_concat['Text'].apply(lambda x: x.count(" ") > 5)) & # Garantir que a sentença não é muito curta
                                     (tweet_df_concat['Text'].apply(lambda x: '@' not in x)), # Não incluir arroba de usuários
                                    'Text']

output_path = 'dataframes/tweet_intensity_filtered.csv'
tweet_df_final.to_csv(output_path, index=False)
