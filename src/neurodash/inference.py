import random
import re
from collections import defaultdict

import pandas as pd
import spacy
from neuradicon.custom_pipes import DomainDetector, SpacySectioner
from spacy.util import minibatch
from tqdm import tqdm


def doc_word_length(doc):
    """Calculate the number of words in spacy doc"""
    return len(doc)


def is_comparitive(doc):
    """return if doc is comparitive"""
    return doc.cats["IS_COMPARATIVE"] > 0.5


def normality_class(doc):
    """return if doc is normal vs abnormal"""
    return max(doc.cats, key=lambda k: doc.cats[k])


def uses_contrast(df):
    contrast_condition = pd.concat(
        [
            df["Procedure"].str.contains("+c", case=False, regex=False),
            df["Procedure"].str.contains("contrast", case=False, regex=False),
            df["Procedure"].str.contains("Post Gad", case=False, regex=False),
            df["Narrative"].str.contains("Post Gad", case=False, regex=False),
            df["Narrative"].str.contains("MR+c", case=False, regex=False),
            df["Narrative"].str.contains("+ Gd", case=False, regex=False),
            df["Narrative"].str.contains("post gadolinium", case=False, regex=False),
        ],
        axis=1,
    ).any(axis=1)
    return contrast_condition


def clean_text(string):
    # Decode/reformat text strings
    new_string = re.sub(r"\\\w+|\{.*?\}|}", "", string)
    new_string = re.sub(r"(\n+)", " ", new_string)
    new_string = re.sub(r"(\r+)", " ", new_string)
    new_string = re.sub(" .", ".", new_string)
    new_string = re.sub(r"_\S+_", "", new_string)
    new_string = re.sub(r"\s{2,}", " ", new_string)
    new_string = new_string.strip()
    return new_string


class DashboardInferenceEngine:
    def __init__(self, model_path_dict, use_gpu=False):
        self.path_dict = model_path_dict
        self.normality_model = None
        self.comparison_model = None
        self.domain_model = None
        self.nlp = None
        self.tokenizer = None
        self.sectioner = None
        if use_gpu:
            spacy.prefer_gpu()
        self.load_models()

    def load_models(self):
        self.normality_model = spacy.load(self.path_dict["normality_cls"])
        self.comparison_model = spacy.load(self.path_dict["comparative_cls"])
        self.nlp = spacy.load(self.path_dict["info_model"])
        self.tokenizer = spacy.load(
            self.path_dict["tokenizer"], exclude=["tagger", "parser", "ner"]
        )
        self.domain_model = DomainDetector(self.nlp)
        self.domain_model.from_disk(self.path_dict["domainer"])
        self.sectioner = SpacySectioner(
            self.path_dict["tokenizer"], self.path_dict["section_cls"]
        )

    def infer_addition_report_data(
        self,
        df,
        infer_data=[
            "uses_contrast",
            "normality_class",
            "is_comparative",
            "report_length_words",
            "sections",
            "pathological_domains",
        ],
        batch_size=128,
        n_processes=1,
    ):
        text_iter = df["Narrative"]

        if "uses_contrast" in infer_data:
            print("inferring contrast")
            df.loc[:, "uses_contrast"] = uses_contrast(df)

        if "normality_class" in infer_data:
            print("inferring normality_class")
            norm_classes = [
                normality_class(doc)
                for doc in tqdm(
                    self.normality_model.pipe(
                        text_iter, batch_size=batch_size, n_process=n_processes
                    )
                )
            ]

            df.loc[:, "normality_class"] = norm_classes

        if "is_comparative" in infer_data:
            print("inferring is_comparitive")
            comp_classes = [
                is_comparitive(doc)
                for doc in tqdm(
                    self.comparison_model.pipe(
                        text_iter, batch_size=batch_size, n_process=n_processes
                    )
                )
            ]
            df.loc[:, "is_comparative"] = comp_classes

        if "report_length_words" in infer_data:
            print("inferring report_word_length")
            lengths = [
                doc_word_length(doc)
                for doc in tqdm(
                    self.nlp.pipe(
                        text_iter, batch_size=batch_size, n_process=n_processes
                    )
                )
            ]

            df.loc[:, "report_length_words"] = lengths

        if "sections" in infer_data:
            print("inferring sections")
            sectioned_reports = self.sectioner(
                text_iter, batch_size=batch_size, n_procs=n_processes
            )
            section_df = pd.DataFrame(sectioned_reports)
            for col in section_df.columns:
                df.loc[:, col] = section_df[col].tolist()

        if "pathological_domains" in infer_data:
            print("inferring pathological domains")
            domains = [
                self.domain_model(doc)._.domains
                for doc in tqdm(
                    self.nlp.pipe(
                        text_iter, batch_size=batch_size, n_process=n_processes
                    )
                )
            ]
            df.loc[:, "pathological_domains"] = domains
            unique_domains = (d for doc in domains for d in doc)
            individual_domains_membership = {
                domain: [domain in doc for doc in domains] for domain in unique_domains
            }
            domain_df = pd.DataFrame.from_dict(individual_domains_membership)
            for col in domain_df.columns:
                df.loc[:, col] = domain_df[col].tolist()
        return df
