

with open('train.txt') as txtfile:
    all_text = txtfile.read()


import csv

with open('train.csv', mode='w') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerow({'text': all_text})


with open('validation.txt') as txtfile:
    all_text = txtfile.read()


import csv

with open('validation.csv', mode='w') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerow({'text': all_text})
