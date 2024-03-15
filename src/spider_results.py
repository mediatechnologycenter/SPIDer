import pickle
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser
from data import MDSData
from mds_pid import MDSPID

def parse_args():
    parser = ArgumentParser()   
    
    parser.add_argument('--dataset', help="Dataset name", default='multi_news')
    parser.add_argument('--max_articles', help="maximum number of articles processed as input", type=int, default='10')

    return parser.parse_args()

def get_stats(df, col):
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()

    return (mean, median, std)

def get_relative_results_df(results, total_sources):
    df = pd.DataFrame(results)[["total_positive_mi", "redundancy", "union", "synergy", "unique"]]
    df["unique"] = df.apply(lambda row: np.array(row["unique"]) / np.array(row["total_positive_mi"]), axis=1)

    df = df.loc[df["total_positive_mi"] > 0]
    df_out = pd.DataFrame(df[["total_positive_mi", "redundancy", "union", "synergy"]].to_numpy() / df[["total_positive_mi"]].to_numpy(),
                      columns=["total_positive_mi", "redundancy", "union", "synergy"], index=df.index)

    df_out["unique"] = df["unique"].apply(lambda x: sum(x)/len(x))
    df_out["unique_variance"] = df["unique"].apply(lambda unique_values: np.var(unique_values))

    df_out["total_sources"] = total_sources
    
    return df_out

def _print_pid_stats(mds_pid_results, file_name, total_sources):
    print(file_name)

    mds_pid_results.print_dataset_prepared_stats()
    print("\n")

    df = get_relative_results_df(mds_pid_results.results, total_sources)

    mean_r, median_r, std_r = get_stats(df, "redundancy")
    mean_u, median_u, std_u = get_stats(df, "union")
    mean_s, median_s, std_s = get_stats(df, "synergy")
    mean_unique, median_unique, std_unique = get_stats(df, "unique")
    mean_unique_var, median_unique_var, std_unique_var = get_stats(df, "unique_variance")

    print(f"Redundancy -- Mean: {mean_r}, Median: {median_r}, Std_dev: {std_r}")
    print(f"Union -- Mean: {mean_u}, Median: {median_u}, Std_dev: {std_u}")
    print(f"Synergy -- Mean: {mean_s}, Median: {median_s}, Std_dev: {std_s}")
    print(f"Unique -- Mean: {mean_unique}, Median: {median_unique}, Std_dev: {std_unique}")
    print(f"Unique variance -- Mean: {mean_unique_var}, Median: {median_unique_var}, Std_dev: {std_unique_var}")

    print("\n")

def _calculate_ranking_of_highest_probability(lists_of_probabilities):
    position_counts = [0]*10
    for probabilities_list in lists_of_probabilities:
        max_index = np.argmax(probabilities_list)
        position_counts[max_index] += 1
       
    total_lists = len(lists_of_probabilities)
    
    ranking = [i/total_lists for i in position_counts]

    return ranking

def print_pid_stats(file_name, total_sources):
    with open(file_name, "rb") as f:
        mds_pid_results = pickle.load(f)        
        _print_pid_stats(mds_pid_results, file_name, total_sources)

def calculate_ranking_of_highest_probability(dataset_name, dataset_df, max_articles):
    ranking = {}
    for total_sources in range(2, max_articles + 1):
        all_unique_lists = dataset_df[dataset_df["total_sources"] == total_sources]["unique"]
        
        if not total_sources in ranking:
            ranking[total_sources] = {}
            
        ranking[total_sources][dataset_name] = _calculate_ranking_of_highest_probability(all_unique_lists)

    return ranking

def main():
    args = parse_args()
    results_path = "../outputs/results/"

    dataset_df = ""
    for total_sources in range(2, args.max_articles + 1):
        file_name = f"{args.dataset}_fixedMDS_dataset__sources_{total_sources}_sample_PID.pkl"
        file_path = f"{results_path}{file_name}"
        print_pid_stats(file_path, total_sources)

        with open(file_path, "rb") as f:
            mds_pid_results = pickle.load(f)

            c_df = get_relative_results_df(mds_pid_results.results, total_sources)
            c_df["dataset"] = args.dataset

            if total_sources == 2:
                dataset_df = c_df
            else:
                dataset_df = pd.concat([dataset_df, c_df], ignore_index=True)

    ranking = calculate_ranking_of_highest_probability(args.dataset, dataset_df, args.max_articles)
    print("Frequency most used article")
    print(ranking)

    sns.boxplot(x='total_sources', y='redundancy', data=dataset_df)
    plt.savefig('redundacy_bloxplot.png')

if __name__ == '__main__':
    main()
