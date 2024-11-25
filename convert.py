import pandas as pd 
import json

#Convert csv to json
df = pd.read_csv('data.csv')
sample_data = df.iloc[0].apply(lambda x: [x]).to_dict()
json_data = json.dumps(sample_data, indent=4) 
print(json_data)
with open('sample_data.json', 'w') as f:
        f.write(json_data)