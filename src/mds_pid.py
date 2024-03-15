from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import re
import numpy as np
import copy
from typing import Any, TypedDict, Iterable
import pickle
import gc


class SampleDict(TypedDict):
    docs: list[list[str]]
    sentences: list[str]
    summary: list[str]
    pmis: list[list[list[float]]]


class MDSPID:
    def __init__(self, dataset_name: str | None = None,
                 dataset: Any | None = None,
                 model_name: str | None = "gpt2-medium",
                 model: Any | None = None,
                 tokenizer: Any | None = None,
                 split: str = None,
                 separator: str = "|||||",
                 batch_size: int = 16,
                 device="cpu",
                 load_model: bool = True) -> None:
        """This class allows to compute the PIDs of a given dataset of documents and
        corresponding summaries.

        Args:
            dataset_name (str | None, optional): the name of the dataset if the desired
            dataset is available on Huggingface. Defaults to None.
            dataset (Any | None, optional): a custom dataset if `dataset_name` is None.
            Defaults to None.
            model_name (str | None, optional): the name of the base LM if available on Huggingface.
            Defaults to "gpt2-medium".
            model (Any | None, optional): a custom LM if `model_name` is None.
            Defaults to None.
            tokenizer (Any | None, optional): a custom tokenizer if `model_name` is None.
            Defaults to None.
            split (str, optional): key of the dataset. Defaults to "train".
            separator (str, optional): the separator between documents. Defaults to "|||||".
            batch_size (int, optional): batch size for probabilities computation. Defaults to 16.
            device (str, optional): device for PyTorch. Defaults to "cpu".
            load_model (bool, optional): whether to load the model (not needed if the dataset_completed
            will be loaded directly (precomputed)). Defaults to True.

        Raises:
            ValueError: Either `dataset_name` or `dataset` has to be specified.
            ValueError: Either `model_name` or (`model` and `tokenizer`) has to be specified.
        """
        self.device = device
        self.separator = separator
        if dataset_name is not None:
            self.dataset = load_dataset(dataset_name, split=split)
        elif dataset is not None:
            self.dataset = dataset
            if split is not None:
                self.dataset = self.dataset[split]
        else:
            raise ValueError(
                "Either `dataset_name` or `dataset` has to be specified.")
        self.dataset_preprocessed = []
        self.dataset_completed = []
        self.split = split
        self.prepared = False

        if load_model:
            if model_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name).to(self.device)
            elif model is not None and tokenizer is not None:
                self.tokenizer = tokenizer
                self.model = model.to(self.device)
            else:
                raise ValueError(
                    "Either `model_name` or (`model` and `tokenizer`) has to be specified.")
        else:
            self.model = None
            self.tokenizer = None

        self.results = []
        self.all_best_collection_redundancy = []
        self.all_best_collection_union = []

        self.batch_size = batch_size

    def _preprocess_sample(self, sample: dict) -> SampleDict:
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

        preprocessed_sample: SampleDict = {
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

    def compute_probs_parallel(self, s_list: list[str], q_list: list[str]) -> tuple[list[list[float]],
                                                                                    list[list[float]],
                                                                                    list[float],
                                                                                    list[float]]:
        """Parallelized way to compute the needed probabilities
        of pairs s and q: pmi(s, q).

        Args:
            s_list (list[str]): list of summary sentences
            q_list (list[str]): list of all documents sentences

        Returns:
            list[list[float]]: pmis
            list[list[float]]: cond_prob
            list[float]: tot_log_prob
            list[float]: tot_log_prob_s
        """
        self.model.eval()
        q_tokenized = []
        q_logits = []
        for q in q_list:
            q_tokenized.append(self.tokenizer.encode(
                q.strip(), padding=True, truncation=True, return_tensors="pt")[0].to(self.device))
            with torch.no_grad():
                logits = self.model(q_tokenized[-1].unsqueeze(0))["logits"]
                logits = torch.log_softmax(logits, dim=-1)
            q_logits.append(logits.to(self.device))
        tot_log_prob = [0 for _ in range(len(q_list))]
        for i_q in range(len(tot_log_prob)):
            for i in range(q_tokenized[i_q].shape[0] - 1):
                tot_log_prob[i_q] += q_logits[i_q][0,
                                                   i, q_tokenized[i_q][i+1]].item()

        first_indices = []
        s_tokenized = []
        s_logits = []
        for s in s_list:
            s_tokenized.append(self.tokenizer.encode(
                s.strip(), padding=True, truncation=True, return_tensors="pt")[0].to(self.device))
            first_indices.append(s_tokenized[-1].shape[0])
            with torch.no_grad():
                logits = self.model(s_tokenized[-1].unsqueeze(0))["logits"]
                logits = torch.log_softmax(logits, dim=-1)
            s_logits.append(logits.to(self.device))
        tot_log_prob_s = [0 for _ in range(len(s_list))]
        for i_s in range(len(tot_log_prob_s)):
            for i in range(s_tokenized[i_s].shape[0] - 1):
                tot_log_prob_s[i_s] += s_logits[i_s][0,
                                                     i, s_tokenized[i_s][i+1]].item()

        current_total_index = 0
        batch_size = self.batch_size
        batched_pairs = []
        for i_s, s in enumerate(s_list):
            for i_q, q in enumerate(q_list):
                if current_total_index == 0 or len(batched_pairs[-1]) >= batch_size:
                    batched_pairs.append([])
                # cond_sent = s + " " + q
                cond_sent = q + " " + s
                batched_pairs[-1].append(cond_sent)
                current_total_index += 1
        pmis = [[]]
        cond_prob = [[]]
        i_s = 0
        i_q = 0
        for batch in batched_pairs:
            gc.collect()
            torch.cuda.empty_cache()
            encodings_dict = self.tokenizer(
                batch, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt')
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)
            last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
            with torch.no_grad():
                logits_cond = self.model(
                    input_ids, attention_mask=attention_mask)["logits"]
                logits_cond = torch.log_softmax(logits_cond, dim=-1)
            for i in range(logits_cond.shape[0]):
                tot_log_prob_cond = 0
                if len(pmis[-1]) >= len(q_list):
                    pmis.append([])
                    cond_prob.append([])
                    i_s += 1
                    i_q = 0
                for j in range(first_indices[i_s], min([input_ids.shape[1] - 1, last_non_masked_idx[i]])):
                    tot_log_prob_cond += logits_cond[i, j,
                                                     input_ids[i, j+1]].item()
                # if self.use_ppmi:
                #     pmis[-1].append(max([0, tot_log_prob_cond -
                #                     tot_log_prob[i_q]]))
                else:
                    pmis[-1].append(tot_log_prob_cond - tot_log_prob[i_q])
                cond_prob[-1].append(tot_log_prob_cond)
                i_q += 1
        return pmis, cond_prob, tot_log_prob, tot_log_prob_s

    def get_all_probs_parallel(self, doc_sents: list[list[str]],
                              sum_sents: list[str]) -> list[list[list[float]]]:
        """Compute the needed probabilities of each pair of document and summary sentence.

        Args:
            doc_sents (list[list[str]]): list of documents sentences list.
            sum_sents (list[str]): list of summary sentences.

        Returns:
            list[list[list[float]]]: pmis
            list[list[list[float]]]: cond_probs
            list[float]: tot_log_prob
            list[float]: tot_log_prob_s
        """
        pmis: list[list[list[float]]] = []
        all_sents = []
        for d in doc_sents:
            all_sents += d
        condensed_pmis, condensed_cond_prob, tot_log_prob, tot_log_prob_s = \
            self.compute_probs_parallel(sum_sents, all_sents)
        cond_prob: list[list[list[float]]] = []
        for i_s, _ in enumerate(sum_sents):
            pmis.append([])
            cond_prob.append([])
            i = 0
            for d in doc_sents:
                pmis[-1].append(condensed_pmis[i_s][i:i+len(d)])
                cond_prob[-1].append(condensed_cond_prob[i_s][i:i+len(d)])
                i += len(d)
        return pmis, cond_prob, tot_log_prob, tot_log_prob_s

    def complete_dataset(self, verbose=True):
        """After having preprocessed the dataset (extracting sentences),
        this method computes all the pmis to then be able to compute PIDs.

        Args:
            verbose (bool, optional): whether to log process. Defaults to True.

        Raises:
            SyntaxError: `preprocess_dataset` should be called before `complete_dataset`.
        """
        if len(self.dataset_preprocessed) == 0:
            raise SyntaxError(
                "`preprocess_dataset` should be called before `complete_dataset`.")
        ds_copy = copy.deepcopy(self.dataset_preprocessed)
        for i, sample in enumerate(ds_copy):
            all_pmis, cond_prob, tot_log_prob, tot_log_prob_s = \
                self.get_all_probs_parallel(
                    sample["docs"], sample["summary"])
            ds_copy[i]["pmis"] = all_pmis
            ds_copy[i]["cond_probs"] = cond_prob
            ds_copy[i]["log_prob_q"] = tot_log_prob
            ds_copy[i]["log_prob_s"] = tot_log_prob_s

            if verbose and (i+1) % 10 == 0:
                print("{:03d}".format(i+1), "/",
                      "{:03d}".format(len(ds_copy)), "done")
        self.dataset_completed = ds_copy

    def get_mutual_info(self, cond_probs: list[list[list[float]]],
                        log_prob_q: list[float],
                        log_prob_s: list[float],
                        i_qs: list[int], sample) -> float:
        """Compute the MI of for the subset i_qs of source sentences.

        Args:
            cond_probs (list[list[list[float]]]): conditional probs of the sample
            log_prob_q (list[float]): log probs of the source sentences
            log_prob_s: list[float]: log probs of the summary sentences
            i_qs: list[int]: indices of the subset of source sentences

        Returns:
            float: MI
        """
        if len(i_qs) == 0:
            return 0
        log_prob_q = np.array(log_prob_q)[i_qs]
        log_prob_s = np.array(log_prob_s)

        log_prob_s_and_q = self.get_completed_matrix_S_x_Q(cond_probs)
        log_prob_s_and_q = log_prob_s_and_q[:, i_qs]

        normalized_log_prob_s_and_q = log_prob_s_and_q - \
            torch.logsumexp(torch.from_numpy(
                log_prob_s_and_q.reshape(-1)), dim=-1).cpu().numpy()

        pmis = normalized_log_prob_s_and_q - \
            torch.log_softmax(torch.from_numpy(
                log_prob_q), dim=-1).cpu().numpy().reshape(1, -1).repeat(normalized_log_prob_s_and_q.shape[0], 0)

        pmis -= torch.log_softmax(torch.from_numpy(log_prob_s),
                                  dim=-1).cpu().numpy().reshape(-1, 1).repeat(pmis.shape[1], 1)

        product_probs = pmis * np.exp(normalized_log_prob_s_and_q)

        # product_probs = product_probs[:, i_qs]

        return max([0., product_probs.sum()])

    def get_mutual_info_by_summary_sentence(self, cond_probs: list[list[list[float]]],
                                            log_prob_q: list[float],
                                            log_prob_s: list[float],
                                            i_qs: list[int], sample) -> np.ndarray:
        """Compute the MI split by summary sentences.

        Args:
            cond_probs (list[list[list[float]]]): conditional probs of the sample
            log_prob_q (list[float]): log probs of the source sentences
            log_prob_s: list[float]: log probs of the summary sentences
            i_qs: list[int]: indices of the subset of source sentences

        Returns:
            np.ndarray: MI by summary sentences
        """
        if len(i_qs) == 0:
            return np.zeros(len(log_prob_s))
        log_prob_q = np.array(log_prob_q)[i_qs]
        log_prob_s = np.array(log_prob_s)

        log_prob_s_and_q = self.get_completed_matrix_S_x_Q(cond_probs)
        log_prob_s_and_q = log_prob_s_and_q[:, i_qs]

        normalized_log_prob_s_and_q = log_prob_s_and_q - \
            torch.logsumexp(torch.from_numpy(
                log_prob_s_and_q), dim=-1).cpu().numpy().reshape(-1, 1).repeat(log_prob_s_and_q.shape[1], 1)

        pmis = normalized_log_prob_s_and_q - \
            torch.log_softmax(torch.from_numpy(
                log_prob_q), dim=-1).cpu().numpy().reshape(1, -1).repeat(normalized_log_prob_s_and_q.shape[0], 0)

        pmis -= torch.log_softmax(torch.from_numpy(log_prob_s),
                                  dim=-1).cpu().numpy().reshape(-1, 1).repeat(pmis.shape[1], 1)  # not sure !!!

        product_probs = pmis * np.exp(normalized_log_prob_s_and_q)

        return np.array([product_probs[i].sum() for i in range(product_probs.shape[0])])

    def get_completed_matrix_S_x_Q(self, cond_probs: list[list[list[float]]]) -> np.ndarray:
        """Reshape the cond_probs to be of dimensions :math:`(|S|, \\sum_i |Q_i|)` where :math:`S` is
        the set of summary sentences and :math:`Q_i` is the set of sentences in document :math:`i`.

        Args:
            cond_probs (list[list[list[float]]]): conditional probabilities

        Returns:
            np.ndarray: reshaped cond_probs
        """
        matrix = []
        for i in range(len(cond_probs)):
            row = []
            for j in range(len(cond_probs[i])):
                row += cond_probs[i][j]
            matrix.append(row)
        return np.array(matrix)
    
    def find_best_collection_redundancy_beam(self, sample: SampleDict, width: int) -> tuple[list, float, float]:
        """Beam search computation of redundancy

        Args:
            sample (SampleDict): sample
            width (int): width of beam search

        Returns:
            tuple[list, float, float]: collection sentence indices, mutual information, mutual information
        """
        sentences_indices = [j for j in range(len(sample["sentences"]))]

        doc_sentences_indices = self.get_docs_sentence_indices(sample)
        docs_mi = [self.get_mutual_info(
            sample["cond_probs"],
            sample["log_prob_q"],
            sample["log_prob_s"], x, sample) for x in doc_sentences_indices]
        min_doc_mi = np.array(docs_mi).min()

        docs_mi_summary_sentences = [self.get_mutual_info_by_summary_sentence(
            sample["cond_probs"],
            sample["log_prob_q"],
            sample["log_prob_s"], x, sample) for x in doc_sentences_indices]
        min_doc_mi_summary_sentences = np.array(
            docs_mi_summary_sentences).min(axis=0)

        current_collection = ([], 0, 0)
        
        searching_tree = [current_collection[0]]

        while len(searching_tree) > 0:
            new_searching_tree = []
            mis = []
            for coll in searching_tree:
                possible_indices = []
                mutual_infos = []
                for i in sentences_indices:
                    if i in coll:
                        continue
                    mi = self.get_mutual_info(sample["cond_probs"],
                                              sample["log_prob_q"],
                                              sample["log_prob_s"],
                                              coll+[i], sample)
                    if mi > min_doc_mi:
                        continue
                    mi_summary_sentences = self.get_mutual_info_by_summary_sentence(sample["cond_probs"],
                                                                                    sample["log_prob_q"],
                                                                                    sample["log_prob_s"],
                                                                                    coll+[i], sample)
                    if (mi_summary_sentences <= min_doc_mi_summary_sentences).all():
                        possible_indices.append(i)
                        mutual_infos.append(mi)

                if len(possible_indices) == 0:
                    # cannot add any sentence without breaking the constraints
                    continue

                # select the largest possible MI wrt given summary sentence
                mutual_infos = np.array(mutual_infos)
                k = min([width, len(mutual_infos)])
                topk_local_indices = np.argpartition(mutual_infos, -k)[-k:]
                topk_indices_sentences = [possible_indices[x] for x in topk_local_indices]
                for j, index in enumerate(topk_indices_sentences):
                    if mutual_infos[j] > current_collection[2]:
                        current_collection = (coll+[index],
                                              mutual_infos[j],
                                              mutual_infos[j])
                    new_searching_tree.append(coll+[index])
                    mis.append(mutual_infos[j])
            if len(new_searching_tree) == 0:
                break
            searching_tree = []
            mis = np.array(mis)
            k = min([width, len(mis)])
            topk_indices = np.argpartition(mis, -k)[-k:]
            for j in topk_indices:
                searching_tree.append(new_searching_tree[j])

        return current_collection

    def find_best_collection_union_beam(self, sample: SampleDict, width: int) -> tuple[list, float, float]:
        """Beam search computation of union information

        Args:
            sample (SampleDict): sample
            width (int): width of beam search

        Returns:
            tuple[list, float, float]: collection sentence indices, MI, MI
            float: total mutual information
        """
        sentences_indices = [j for j in range(len(sample["sentences"]))]

        doc_sentences_indices = self.get_docs_sentence_indices(sample)
        docs_mi = [self.get_mutual_info(
            sample["cond_probs"],
            sample["log_prob_q"],
            sample["log_prob_s"], x, sample) for x in doc_sentences_indices]
        max_doc_mi = np.array(docs_mi).max()

        docs_mi_summary_sentences = [self.get_mutual_info_by_summary_sentence(
            sample["cond_probs"],
            sample["log_prob_q"],
            sample["log_prob_s"], x, sample) for x in doc_sentences_indices]
        max_doc_mi_summary_sentences = np.array(
            docs_mi_summary_sentences).max(axis=0)

        total_mutual_info = self.get_mutual_info(sample["cond_probs"],
                                                 sample["log_prob_q"],
                                                 sample["log_prob_s"],
                                                 list(sentences_indices), sample)

        current_collection = (list(sentences_indices), total_mutual_info,
                              total_mutual_info)
        
        searching_tree = [current_collection[0]]

        while len(searching_tree) > 0:
            new_searching_tree = []
            mis = []
            for coll in searching_tree:
                possible_indices = []
                mutual_infos = []
                for i in coll:
                    new_coll = [k for k in coll if k != i]
                    mi = self.get_mutual_info(sample["cond_probs"],
                                              sample["log_prob_q"],
                                              sample["log_prob_s"],
                                              new_coll, sample)
                    if mi < max_doc_mi:
                        continue
                    mi_summary_sentences = self.get_mutual_info_by_summary_sentence(sample["cond_probs"],
                                                                                    sample["log_prob_q"],
                                                                                    sample["log_prob_s"],
                                                                                    new_coll, sample)
                    if (mi_summary_sentences < max_doc_mi_summary_sentences).any():
                        continue
                    
                    possible_indices.append(i)
                    mutual_infos.append(mi)

                if len(possible_indices) == 0:
                    # cannot add any sentence without breaking the constraints
                    continue

                # select the largest possible MI wrt given summary sentence
                mutual_infos = -np.array(mutual_infos)
                k = min([width, len(mutual_infos)])
                topk_local_indices = np.argpartition(mutual_infos, -k)[-k:]
                topk_indices_sentences = [possible_indices[x] for x in topk_local_indices]
                mutual_infos = -mutual_infos
                for j, index in enumerate(topk_indices_sentences):
                    new_coll = [m for m in coll if m != index]
                    if mutual_infos[j] < current_collection[2]:
                        current_collection = (new_coll,
                                              mutual_infos[j],
                                              mutual_infos[j])
                    new_searching_tree.append(new_coll)
                    mis.append(mutual_infos[j])
            if len(new_searching_tree) == 0:
                break
            searching_tree = []
            mis = np.array(mis)
            k = min([width, len(mis)])
            mis = -mis
            topk_indices = np.argpartition(mis, -k)[-k:]
            for j in topk_indices:
                searching_tree.append(new_searching_tree[j])

        return current_collection, total_mutual_info

    def prepare_dataset(self, indices: Iterable = None, verbose=True):
        """Prepare the dataset for PIDs computation (extract sentences and compute
        pmis).

        Args:
            indices (Iterable, optional): sample indices of the dataset to prepare. Defaults to None.
            verbose (bool, optional): whether to log process. Defaults to True.
        """
        self.preprocess_dataset(indices=indices, verbose=verbose)
        self.complete_dataset(verbose)
        self.prepared = True

    def load_prepared_dataset(self, path_file: str = "dataset_completed.pkl"):
        """Load an already prepared dataset.

        Args:
            path_file (str, optional): path to the prepared dataset. Defaults to "dataset_completed.pkl".
        """
        with open(path_file, "rb") as f:
            self.dataset_completed = pickle.load(f)
        self.prepared = True

    def get_mean_max_min(self, key: str):
        return np.array([len(x[key]) for x in self.dataset_completed]).mean(), \
            np.array([len(x[key]) for x in self.dataset_completed]).max(), \
            np.array([len(x[key]) for x in self.dataset_completed]).min()

    def print_dataset_prepared_stats(self):
        """Log stats about the prepared dataset.
        """
        if not self.prepared:
            print("Not completed yet.")
        mean_sents, max_sents, min_sents = self.get_mean_max_min("sentences")
        mean_docs, max_docs, min_docs = self.get_mean_max_min("docs")
        mean_sum, max_sum, min_sum = self.get_mean_max_min("summary")
        print(f"Number of samples: {len(self.dataset_completed)}")
        print("=" * len(f"Number of samples: {len(self.dataset_completed)}"))
        print("Documents sentences")
        print("Mean:", "{:.2f}".format(mean_sents),
              "Max:", "{}".format(max_sents),
              "Min:", "{}".format(min_sents))
        print("Documents")
        print("Mean:", "{:.2f}".format(mean_docs),
              "Max:", "{}".format(max_docs),
              "Min:", "{}".format(min_docs))
        print("Summary sentences")
        print("Mean:", "{:.2f}".format(mean_sum),
              "Max:", "{}".format(max_sum),
              "Min:", "{}".format(min_sum))

    def prepare_sample(self, sample):
        """Normalize the sentences probabilities by their length

        Args:
            sample (dict): sample

        Returns:
            dict: updated sample
        """
        cond_probs = []
        for i_s, s in enumerate(sample["cond_probs"]):
            cond_probs.append([])
            i_tot_q = 0
            for i_d, d in enumerate(s):
                cond_probs[-1].append([])
                for i_q, q in enumerate(d):
                    le = len(sample["docs"][i_d][i_q].split()) + len(sample["summary"][i_s].split())
                    cond_probs[-1][-1].append((q + sample["log_prob_q"][i_tot_q]) / le)
                    i_tot_q += 1
        log_prob_s = []
        for i_s, s in enumerate(sample["log_prob_s"]):
            le = len(sample["summary"][i_s].split())
            log_prob_s.append(s / le)
        log_prob_q = []
        for i_q, q in enumerate(sample["log_prob_q"]):
            le = len(sample["sentences"][i_q].split())
            log_prob_q.append(q / le)
        return {**sample, "cond_probs": cond_probs, "log_prob_s": log_prob_s, "log_prob_q": log_prob_q}

    def compute_PIDs(self, indices: Iterable = None, beam_width: int = 4,
                     verbosity=10) -> tuple[list[dict], list[tuple], list[tuple]]:
        """Compute the PID for each sample in the prepared dataset according to indices.

        Args:
            indices (Iterable, optional): sample indices to compute PID for. Defaults to None.
            verbosity (int, optional): log process. Defaults to 10.

        Raises:
            SyntaxError: `prepare_dataset` or `load_prepared_dataset` should be called before.

        Returns:
            tuple[list[dict], list[tuple], list[tuple]]: results PIDs, redundancy collections, union collections
        """
        if not self.prepared:
            raise SyntaxError(
                "`prepare_dataset` or `load_prepared_dataset` should be called before.")
        reslts = []
        all_best_collection_union = []
        all_best_collection_redundancy = []
        if indices is None:
            indices = range(len(self.dataset_completed))
        for step, k in enumerate(indices):
            sample = self.dataset_completed[k]
            sample = self.prepare_sample(sample)
            # Redundancy
            best_collection_redundancy = self.find_best_collection_redundancy_beam(
                sample, beam_width)

            # Union information
            best_collection_union, total_positive_mi = \
                self.find_best_collection_union_beam(sample, beam_width)

            res_sample = {
                "redundancy": best_collection_redundancy[2],
                "union": min([total_positive_mi, best_collection_union[2]]),
                "total_positive_mi": total_positive_mi
            }

            synergy = total_positive_mi - res_sample["union"]
            res_sample["synergy"] = synergy

            unique_infos = []
            sentences_indices = self.get_docs_sentence_indices(sample)
            docs_mi = [self.get_mutual_info(
                sample["cond_probs"],
                sample["log_prob_q"],
                sample["log_prob_s"], x, sample) for x in sentences_indices]

            # avoid numerical errors
            res_sample["redundancy"] = min(
                [np.array(docs_mi).min(), res_sample["redundancy"]])
            best_collection_redundancy = (best_collection_redundancy[0], res_sample["redundancy"],
                                          res_sample["redundancy"])

            for mi in docs_mi:
                unique_infos.append(
                    min([mi, total_positive_mi]) - res_sample["redundancy"])
            res_sample["unique"] = unique_infos

            reslts.append(res_sample)
            all_best_collection_redundancy.append(best_collection_redundancy)
            all_best_collection_union.append(best_collection_union)

            if verbosity > 0:
                if (step+1) % verbosity == 0 or (step+1) == len(indices):
                    print("Summary", step+1, "/", len(indices), "done.")

        self.results = reslts
        self.all_best_collection_redundancy = all_best_collection_redundancy
        self.all_best_collection_union = all_best_collection_union
        return reslts, all_best_collection_redundancy, all_best_collection_union

    def save_results(self, save_path: str):
        """Save computed results to the given location.

        Args:
            save_path (str): location
        """
        with open(save_path, "wb") as f:
            pickle.dump([self.results,
                         self.all_best_collection_redundancy,
                         self.all_best_collection_union], f)

    def print_result(self, idx: int):
        """Log the selected sentences for redundancy and union information as well as summary
        sentences.

        Args:
            idx (int): index of the sample to log
        """
        print("====== Redundancy sentences ======")
        pmi_sentence_indices = self.get_pmi_sentence_indices(
            self.dataset_completed[idx]["pmis"])
        for i, q in enumerate(self.dataset_completed[idx]["sentences"]):
            if i in self.all_best_collection_redundancy[idx][0]:
                print(q, pmi_sentence_indices[i])

        print("\n====== Union information sentences ======")
        for i, q in enumerate(self.dataset_completed[idx]["sentences"]):
            if i in self.all_best_collection_union[idx][0]:
                print(q, pmi_sentence_indices[i])

        print("\n====== Summary sentences ======")
        for s in self.dataset_completed[idx]["summary"]:
            print(s)

    def extract_results_sentences(self, results: list[dict], all_best_redundancy: list[tuple],
                                  all_best_union: list[tuple]) -> list[dict]:
        assert len(all_best_redundancy) == len(results)
        assert len(all_best_redundancy) == len(all_best_union)
        assert len(all_best_redundancy) <= len(self.dataset_completed)
        extracted_results = []
        for i, collection in enumerate(all_best_redundancy):
            output = {
                "documents": [],
                "summary": "",
                "redundancy_sentences": [],
                "union_sentences": [],
                "computed_values": results[i]
            }
            output["documents"] = [
                " ".join(x) for x in self.dataset_completed[i]["docs"]]
            output["summary"] = " ".join(self.dataset_completed[i]["summary"])
            pmi_sentences_indices = self.get_pmi_sentence_indices(
                self.dataset_completed[i]["pmis"])
            for j in collection[0]:
                output["redundancy_sentences"].append(
                    self.dataset_completed[i]["sentences"][j] + f" | doc_{pmi_sentences_indices[j][0]}")
            for j in all_best_union[i][0]:
                output["union_sentences"].append(
                    self.dataset_completed[i]["sentences"][j] + f" | doc_{pmi_sentences_indices[j][0]}")
            extracted_results.append(output)
        return extracted_results

    @staticmethod
    def print_extracted_results_sentences(extracted_results: list[dict]):
        for s, output in enumerate(extracted_results):
            print(f"====== SAMPLE {s:03d} ======")
            print("=== Documents ===")
            for i, doc in enumerate(output["documents"]):
                print("Document", i+1)
                print(doc)
            print("\n=== Summary ===")
            print(output["summary"])
            print("\n=== Redundancy sentences ===")
            for i, sent in enumerate(output["redundancy_sentences"]):
                print(f"{i+1}:", sent)
            print("\n=== Union sentences ===")
            for i, sent in enumerate(output["union_sentences"]):
                print(f"{i+1}:", sent)
            print("\n=== Computed valued ===")
            for k in output["computed_values"]:
                if k == "unique":
                    print(k, [float("{:.5f}".format(x))
                          for x in output["computed_values"][k]])
                    continue
                print(k, "{:.5f}".format(output["computed_values"][k]))
            print("="*80)
            print("="*80)

    @staticmethod
    def extracted_results_sentences_to_file(extracted_results: list[dict], filename: str):
        with open(filename, "w") as f:
            lines = []
            for s, output in enumerate(extracted_results):
                lines.append(f"====== SAMPLE {s:03d} ======\n")
                lines.append("=== Documents ===\n")
                for i, doc in enumerate(output["documents"]):
                    lines.append("Document " + str(i+1) + "\n")
                    lines.append(doc + "\n")
                lines.append("\n=== Summary ===\n")
                lines.append(output["summary"] + "\n")
                lines.append("\n=== Redundancy sentences ===\n")
                for i, sent in enumerate(output["redundancy_sentences"]):
                    lines.append(f"{i+1}: " + sent + "\n")
                lines.append("\n=== Union sentences ===\n")
                for i, sent in enumerate(output["union_sentences"]):
                    lines.append(f"{i+1}: " + sent + "\n")
                lines.append("\n=== Computed valued ===\n")
                for k in output["computed_values"]:
                    if k == "unique":
                        lines.append(k + " " + str([float("{:.5f}".format(x))
                            for x in output["computed_values"][k]]) + "\n")
                        continue
                    lines.append(k + " " + "{:.5f}".format(output["computed_values"][k]) + "\n")
                lines.append("="*80 + "\n")
                lines.append("="*80 + "\n")
            f.writelines(lines)

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

    @staticmethod
    def convert_to_numpy(input_list: list[list]) -> np.ndarray:
        """Convert a list of sublist (sublists of different lengths) to a
        numpy array padding with 0's.

        Args:
            input_list (list[list]): input

        Returns:
            np.ndarray: padded matrix
        """
        # Find the length of the longest inner list
        max_length = max(len(sublist) for sublist in input_list)

        # Create a list of numpy arrays with padded values
        padded_arrays = [np.pad(
            sublist, (0, max_length - len(sublist)), 'constant') for sublist in input_list]

        # Convert the list of arrays into a numpy array
        numpy_array = np.array(padded_arrays)

        return numpy_array

    @staticmethod
    def get_pmi_sentence_indices(all_pmis: list[list[list[float]]]) -> list[tuple[int, int]]:
        """Get the tuple indices (document index, in document sentence index).

        Args:
            all_pmis (list[list[list[float]]]): pmis

        Returns:
            list[tuple[int, int]]: tuple indices
        """
        indices = []
        for j in range(len(all_pmis[0])):
            for k in range(len(all_pmis[0][j])):
                indices.append((j, k))
        return indices

    @staticmethod
    def get_docs_sentence_indices(sample):
        doc_sentences_indices = []
        q = 0
        for d in sample["cond_probs"][0]:
            doc_sentences_indices.append([])
            for _ in d:
                doc_sentences_indices[-1].append(q)
                q += 1
        return doc_sentences_indices

    @staticmethod
    def check_redundancy(res):
        for i, x in enumerate(res):
            if x["redundancy"] < 0:
                print("Redundancy smaller than 0.")
                print(i, x)
            if (np.array(x["unique"]) < 0).any():
                print("Unique smaller than 0.")
                print(i, x)

    @staticmethod
    def check_union(res):
        for i, x in enumerate(res):
            if not (x["union"] * np.ones(len(x["unique"])) -
                           np.array(x["unique"]) + x["redundancy"] > -1e-5).all():
                print("Not all source MI are smaller than union.")
                print(i, x)
            if x["union"] > x["total_positive_mi"]:
                print("Union larger than total positive MI.")
                print(i, x)
            if x["union"] < 0:
                print("Union smaller than 0.")
                print(i, x)
            if x["synergy"] < 0:
                print("Synergy smaller than 0.")
                print(i, x)
