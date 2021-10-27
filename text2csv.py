import csv
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="Folder of texts")
parser.add_argument('--split', action='store_true')
args = vars(parser.parse_args())


def folder2csv():
    if not args['split']:
        args['split'] = 80
    elif (args['split'] > 100 or args['split'] <= 0):
        return 'split must be between 100 and 0'



    folder_path = args['f']
    files_list = os.listdir(folder_path)
    random.shuffle(files_list)

    num_files = len(files_list)
    num_files = int(num_files * (args['split']/100))
    train_files = files_list[:num_files]
    valid_files = files_list[num_files:]

    # Train
    with open('train.csv', mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for file in train_files:
        with open(folder_path+'/'+file, encoding='utf-8') as txtfile:
            all_text = txtfile.read()
            all_text = all_text + "\n<|endoftext|>"

        with open('train.csv', mode='a', encoding='utf-8', newline='\n') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([all_text])



    # VALID
    with open('validation.csv', mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for file in valid_files:
        with open(folder_path+'/'+file, encoding='utf-8') as txtfile:
            all_text = txtfile.read()
            all_text = all_text + "\n<|endoftext|>"

        with open('validation.csv', mode='a', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([all_text])

    print("created train.csv and validation.csv files")


def text2csv():
    with open('train.txt', encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open('train.csv', mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})


    with open('validation.txt', encoding='utf-8') as txtfile:
        all_text = txtfile.read()
    with open('validation.csv', mode='w', encoding='utf-8') as csv_file:
        fieldnames = ['text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'text': all_text})

    print("created train.csv and validation.csv files")


if args['f']:
    folder2csv()
else:
    text2csv()

