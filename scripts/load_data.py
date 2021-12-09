import pickle as pck
import csv

with open('../pck/cl_data.pck', 'rb') as f:
    data = pck.load(f)

with open('../csv/data.csv', 'w') as f:
    writer = csv.writer(f)
    for row in data:
        if type(row) != str:
            writer.writerow(row)
