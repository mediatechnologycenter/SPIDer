import pickle
import torch

from utils import parse_args, ModelConfig, GeneralConfig, DataConfig
from data import MDSData
from mds_pid import MDSPID

def pid_sanity_checks(mdspid, results):
    mdspid.check_redundancy(results)
    mdspid.check_union(results)

    count = 0
    for res in results:
        count += res["redundancy"] == 0
    count, len(results)

def prepare_data(data_config, experiment_key, suffix):
    # Prepare Data
    mds_data = MDSData(dataset_name=data_config.dataset_name, split=data_config.split)
    mds_data.create_fixedMDS_dataset(total_sources=data_config.total_sources, random_sample=data_config.random_sample)

    mds_data.save_dataset(key=experiment_key, data_path="../outputs/preprocessed_data/", suffix=suffix)

def compute_probabilities(dataset_file_path, model_id, device, batch_size, prob_file_path):
    with open(dataset_file_path, "rb") as f:
        dataset = pickle.load(f)

    # Compute Mutual Information Probabilities
    # Note that to use a custom dataset you need to set `dataset_name` to None and pass the `dataset` argument
    mdspid = MDSPID(dataset_name=None,
                model_name=model_id,
                dataset=dataset,
                split=None,
                device=device,
                batch_size=batch_size if device == "cuda" else 1)
    
    mdspid.prepare_dataset()

    total_samples = str(len(mdspid.dataset))
    print("total samples in processed dataset:", total_samples)
    mdspid.print_dataset_prepared_stats()

    with open(prob_file_path, "wb") as f:
        pickle.dump(mdspid, f)
    
def compute_pid(prob_file_path, pid_file_path):
    # Compute Partial Information Decompostion
    with open(prob_file_path, "rb") as f:
        mdspid = pickle.load(f)

    results, _, _ = mdspid.compute_PIDs()
    pid_sanity_checks(mdspid, results)

    with open(pid_file_path, "wb") as f:
        pickle.dump(mdspid, f)

def extract_sentences(pid_file_path, extracted_sentences_path):
    with open(pid_file_path, "rb") as f:
        mdspid = pickle.load(f)

    extracted_results_sentences = mdspid.extract_results_sentences(mdspid.results, mdspid.all_best_collection_redundancy, mdspid.all_best_collection_union)
    mdspid.extracted_results_sentences_to_file(extracted_results_sentences, extracted_sentences_path)

def file_name(data_config, experiment_key):
    output_name = data_config.dataset_name+"_"+experiment_key
    suffix = "_sources_"+str(data_config.total_sources)

    if data_config.random_sample:
        suffix += "_sample"

    return (output_name, suffix)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    
    model_config = ModelConfig(**args.model_config)
    general_config = GeneralConfig(**args.general_config)
    data_config = DataConfig(**args.data_config)

    output_name, suffix = file_name(data_config, general_config.experiment_key)
    dataset_file_path = f"../outputs/preprocessed_data/{output_name}_{suffix}.pkl"
    prob_file_path = f"../outputs/precomputed/{output_name}_{suffix}_MI.pkl"
    pid_file_path = f"../outputs/results/{output_name}_{suffix}_PID.pkl"
    extracted_sentences_path = f"../outputs/results/{output_name}_{suffix}_extracted_sentences.txt"

    if args.mode == 'run_all':
        prepare_data(data_config, general_config.experiment_key, suffix=output_name+"_"+suffix)

    if args.mode in ['run_all', 'run_prob']:
        compute_probabilities(dataset_file_path, model_config.model_id, device, general_config.batch_size, prob_file_path)

    if args.mode in ['run_all', 'run_prob', 'run_pid']:
        # Compute Partial Information Decompostion
        compute_pid(prob_file_path, pid_file_path)

    # Extract sentences
    extract_sentences(pid_file_path, extracted_sentences_path)


if __name__ == '__main__':
    main()
