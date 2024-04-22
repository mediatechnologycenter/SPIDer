
# SPIDer: Summarization Analysis with Partial Information Decomposition
<img src="spider_logo.png" alt="SPIDer Logo" width="130" height="auto">

## Description

SPIDer is a framework that decomposes the mutual information contained in a multi-document summary into redundant, synergistic, union, and unique information. We base our implementation on the Partial Information Decomposition (PID) approach defined in [A Novel Approach to the Partial Information Decomposition](https://www.mdpi.com/1099-4300/24/3/403).

## Installation

Create a Python environment and run the following command to install the required packages:

```
pip install -r requirements.txt
```

## Usage

The class `MDSPID` works as follows:
1. define the dataset and underlying language model you want to use
2. compute the needed sentence probabilities
3. run the computation of PIDs (partial information decomposition)

Important: Create the output folders ```outputs/precomputed```, ```outputs/preprocessed_data```, and ```outputs/results```

Run: 
```
python run_pid.py --mode run_all --config ../configs/config.yaml
```

After running ```run_spider.py```, the `MDSPID` instance with the computed PID is stored as a pickle file under `outputs/results`. To analyze the results you can use the Jupyter Notebook `notebooks/pid_results.ipynb` or the script ```spider_results.py```.

The code to convert MultiRC into a MDS dataset and get the synergy scores is under `notebooks/multiRC`.

## Citation

```bibtex
@inproceedings{TODO,
    title = "",
    author = "",
    booktitle = "",
    month = ,
    year = "",
    address = "",
    publisher = "",
    url = "",
    pages = "",
    abstract = "",
}
```
