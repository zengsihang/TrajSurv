import numpy as np
import pandas as pd
from collections import Counter
import pickle
import argparse

def get_top_labs(df, top_k=10):
    # count the number of times each lab appears
    # the labname is the key and the value is the number of times it appears
    lab_counts = Counter(df.labname)
    print("Finished counting lab names")
    # sort the labs by the number of times they appear
    top_labs = sorted(lab_counts, key=lab_counts.get, reverse=True)
    print("Finished sorting lab names")
    return top_labs[:top_k]

def main(args):
    # load the lab data
    print('Loading lab data...')
    lab_data = pd.read_csv(args.lab_file)
    # get the top labs
    print('Getting top labs...')
    top_labs = get_top_labs(lab_data, top_k=args.top_k)
    print(top_labs)
    # save the top labs
    with open(args.output_file, 'wb') as f:
        pickle.dump(top_labs, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab_file', type=str, required=True, help='Path to eICU lab.csv file')
    parser.add_argument('--top_k', type=int, default=40, help='Number of top labs to extract')
    parser.add_argument('--output_file', type=str, default='top_labs.pkl', help='Output pickle file')
    args = parser.parse_args()
    main(args)
