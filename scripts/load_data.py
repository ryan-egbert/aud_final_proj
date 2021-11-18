import pickle as pck
import csv

def load_data(path):
    with open(path, 'rb') as f:
        data = pck.load(f)
    return data

data = load_data('aud/aud_final_proj/data/cl_data.pck')

with open('data_csv.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(data[:10])