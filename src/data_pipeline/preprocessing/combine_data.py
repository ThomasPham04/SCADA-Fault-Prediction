from data_pipeline.utils.io import *

data_path = r"D:\Final Project\scada-fault-prediction\Dataset\raw\Wind Farm A\datasets" 
file_name = "og_combined_data.csv"
df = read_and_concat_csv(data_path)

save_to_csv(df, file_name, data_path)
