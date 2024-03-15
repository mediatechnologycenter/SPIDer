import numpy as np
import pickle
from typing import Iterable
from datasets import load_dataset
from random import sample as rand_sample
import re


class MDSData:
    def __init__(self, dataset_name: str | None = "multi_news",
                 split: str = None,
                 dataset=None,
                 separator: str = "|||||"):
        if dataset_name is not None:
            self.dataset = load_dataset(dataset_name, split=split)
        elif dataset is not None:
            self.dataset = dataset
            if split is not None:
                self.dataset = self.dataset[split]
        else:
            raise ValueError(
                "Either `dataset_name` or `dataset` has to be specified.")
        self.separator = separator
        self.dataset_preprocessed = []
        self.preprocessed = False

        self.generated_datasets = {
            "fixedMDS_dataset": [], 
            "samedocs_dataset": [],
            "commonsentences_dataset": [], 
            "sparseinfo_dataset": [], 
            "singledoc_dataset": [],
            "hallucination_dataset": []
        }

    def _preprocess_sample(self, sample: dict):
        """Preprocess a sample spliting the texts into sentences.

        Args:
            sample (dict): raw sample from the dataset

        Returns:
            SampleDict: preprocessed sample
        """
        # Source documents
        doc_sentences = []
        all_docs_sentences = []
        docs_split = sample["document"].split(self.separator)
        for doc in docs_split:
            paragraph_split = doc.replace(
                "\n \n", "\n").replace("\n\n", "\n").split("\n")
            sentences = self.split_sentences(paragraph_split)
            doc_sentences.append(sentences)
            all_docs_sentences += sentences
        # Summary
        all_summary_sentences = []
        summary = sample["summary"]
        if summary[:2] == "– ":
            summary = summary[2:]
        paragraph_split_summary = summary.replace(
            "\n \n", "\n").replace("\n\n", "\n").split("\n")
        sentences_summary = self.split_sentences(paragraph_split_summary)
        all_summary_sentences += sentences_summary

        preprocessed_sample = {
            "docs": doc_sentences,
            "sentences": all_docs_sentences,
            "summary": all_summary_sentences
        }
        return preprocessed_sample

    def preprocess_dataset(self, indices: Iterable = None, verbose=True):
        """Preprocess the dataset at given indices.

        Args:
            indices (Iterable, optional): the indices of the samples in the dataset
            to preprocess. None preprocess all samples. Defaults to None.
            verbose (bool, optional): whether to log the process. Defaults to True.
        """
        # for now the structure of the samples is:
        # {"document": str (documents split with `separator`), "summary": str}
        assert "document" in self.dataset[0] and \
            "summary" in self.dataset[0]

        if indices is None:
            indices = range(len(self.dataset))

        dataset_preprocessed = []
        for i in indices:
            dataset_preprocessed.append(
                self._preprocess_sample(self.dataset[i]))

        self.dataset_preprocessed = dataset_preprocessed

        if verbose:
            print(len(dataset_preprocessed),
                  f"sample{'s' if len(dataset_preprocessed) > 1 else ''} preprocessed.")
        self.preprocessed = True

    def create_fixedMDS_dataset(self, total_sources=2, random_sample=True, indices=None, verbose=True):
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)

        fixedMDS_dataset = []
        for entry in self.dataset_preprocessed:
            if total_sources == -1 or len(entry["docs"]) == total_sources:
                sample = {
                    "document": self.separator.join([" ".join(doc) for doc in entry["docs"]]),
                    "summary": " ".join(entry["summary"]),
                }
                fixedMDS_dataset.append(sample)

        if random_sample:
            sample_size = min(len(fixedMDS_dataset), 100)
            self.generated_datasets["fixedMDS_dataset"] = rand_sample(fixedMDS_dataset, sample_size)
        else:
            self.generated_datasets["fixedMDS_dataset"] = fixedMDS_dataset

        print(len(self.generated_datasets["fixedMDS_dataset"]), "samples created.")   

    def create_samedocs_dataset(self, indices=None, verbose=True):        
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)
        np.random.seed(0)
        samedocs_dataset = []
        for i in range(len(self.dataset_preprocessed)):
            if i not in indices:
                continue
            n_docs = np.random.randint(2, 4)
            selected_doc = np.random.randint(
                len(self.dataset_preprocessed[i]["docs"]))
            n_sent_summary = np.random.randint(
                1, min([5, len(self.dataset_preprocessed[i]["docs"][selected_doc])+1]))
            sent_summary_indices = np.random.choice(len(
                self.dataset_preprocessed[i]["docs"][selected_doc]), n_sent_summary, replace=False)
            summary_sentences = [self.dataset_preprocessed[i]
                                 ["docs"][selected_doc][s] for s in sent_summary_indices]
            sample = {
                "document": self.separator.join([" ".join(self.dataset_preprocessed[i]["docs"][selected_doc])]*n_docs),
                "summary": " ".join(summary_sentences),
            }
            samedocs_dataset.append(sample)
        self.generated_datasets["samedocs_dataset"] = samedocs_dataset
        print(len(samedocs_dataset), "samples created.")

    def create_commonsentences_dataset(self, indices=None, verbose=True):
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)
        np.random.seed(0)
        commonsentences_dataset = []
        for i in range(len(self.dataset_preprocessed)):
            if i not in indices:
                continue
            common_sentences = []
            for i_doc, doc in enumerate(self.dataset_preprocessed[i]["docs"]):
                selected_sentence = np.random.randint(len(doc))
                common_sentences.append((i_doc, doc[selected_sentence]))
            docs = self.dataset_preprocessed[i]["docs"]
            for i_doc in range(len(docs)):
                for j_doc, sent in common_sentences:
                    if i_doc == j_doc:
                        continue
                    docs[i_doc].append(sent)
            sample = {
                "document": self.separator.join([" ".join(doc) for doc in docs]),
                "summary": " ".join(self.dataset_preprocessed[i]["summary"]),
                "common_sentences": common_sentences
            }
            commonsentences_dataset.append(sample)
        self.generated_datasets["commonsentences_dataset"] = commonsentences_dataset
        print(len(commonsentences_dataset), "samples created.")

    def create_sparseinfo_dataset(self, indices=None, verbose=True):
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)
        np.random.seed(0)
        sparseinfo_dataset = []
        for i in range(len(self.dataset_preprocessed)):
            if i not in indices:
                continue
            docs = self.dataset_preprocessed[i]["docs"]
            for i_doc in range(len(docs)):
                subset = np.random.choice(
                    len(docs[i_doc]), len(docs[i_doc])//2, replace=False)
                new_sents = []
                for j in subset:
                    new_sents.append(docs[i_doc][j])
                docs[i_doc] = new_sents
            sample = {
                "document": self.separator.join([" ".join(doc) for doc in docs]),
                "summary": " ".join(self.dataset_preprocessed[i]["summary"]),
            }
            sparseinfo_dataset.append(sample)
        self.generated_datasets["sparseinfo_dataset"] = sparseinfo_dataset
        print(len(sparseinfo_dataset), "samples created.")

    def create_singledoc_dataset(self, indices=None, verbose=True):
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)
        np.random.seed(0)
        singledoc_dataset = []
        for i in range(len(self.dataset_preprocessed)):
            if i not in indices:
                continue
            selected_doc = np.random.randint(
                len(self.dataset_preprocessed[i]["docs"]))
            sample = {
                "document": " ".join(self.dataset_preprocessed[i]["docs"][selected_doc]),
                "summary": " ".join(self.dataset_preprocessed[i]["summary"])
            }
            singledoc_dataset.append(sample)
        self.generated_datasets["singledoc_dataset"] = singledoc_dataset
        print(len(singledoc_dataset), "samples created.")

    def create_hallucination_dataset(self, indices=None, verbose=True):
        if not self.preprocessed:
            self.preprocess_dataset(indices=indices, verbose=verbose)
        np.random.seed(0)
        hallucinated_sentences = []
        for i in range(len(self.dataset_preprocessed)):
            selected_sent = np.random.randint(
                len(self.dataset_preprocessed[i]["summary"]))
            hallucinated_sentences.append(
                self.dataset_preprocessed[i]["summary"][selected_sent])
        hallucination_dataset = []
        for i in range(len(self.dataset_preprocessed)):
            if i not in indices:
                continue
            docs = self.dataset_preprocessed[i]["docs"]
            selected_hallucination = np.random.randint(
                len(hallucinated_sentences))
            while selected_hallucination == i:
                selected_hallucination = np.random.randint(
                    len(hallucinated_sentences))
            sample = {
                "document": self.separator.join([" ".join(doc) for doc in docs]),
                "summary": " ".join(self.dataset_preprocessed[i]["summary"] +
                                    [hallucinated_sentences[selected_hallucination]]),
                "hallucinated_sentence": hallucinated_sentences[selected_hallucination]
            }
            hallucination_dataset.append(sample)
        self.generated_datasets["hallucination_dataset"] = hallucination_dataset
        print(len(hallucination_dataset), "samples created.")

    def save_dataset(self, key: str, data_path: str = "../data/experiments/", suffix: str = None):
        #if suffix is None:
        #    suffix = ""
        #else:
        #    suffix = "_" + suffix
        with open(f"{data_path}{suffix}.pkl", "wb") as f:
            pickle.dump(self.generated_datasets[key], f)

    @staticmethod
    def split_sentences(text_list: list[str]) -> list[str]:
        """Split the given list of texts into sentences.

        Args:
            text_list (list[str]): list of texts

        Returns:
            list[str]: list of sentences
        """
        sentences = []
        for text in text_list:
            # Split text into sentences using regular expressions
            sentence_list = re.split(
                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)[\s("\s)]', text)
            # Remove leading/trailing spaces and add each sentence to the result list
            sentences.extend(sentence.strip() + ('"' if sentence.count('"') % 2 != 0 else "")
                             for sentence in sentence_list if sentence.strip())
        return sentences
