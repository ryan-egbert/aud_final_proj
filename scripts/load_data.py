import pickle as pck
import csv

with open('aud/aud_final_proj/data/cl_data.pck', 'rb') as f:
    data = pck.load(f)

with open('aud/aud_final_proj/csv/data.csv', 'w') as f:
    writer = csv.writer(f)
    for row in data:
        if type(row) != str:
            writer.writerow(row)