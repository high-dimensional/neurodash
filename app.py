""" neuroDash dashboard app

Dashboard displaying the selection,
summarization and plotting of
neuroradiological reports using the
neuroNLP package.
"""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import srsly
import streamlit as st
from neuradicon.custom_pipes import (DomainDetector, NegationDetector,
                                     RelationExtractor, SpacySectioner)

from neurodash.clinical import *
from neurodash.operational import *
from neurodash.service_analysis import *
from neurodash.utils import *

SPACY_INFO_MODEL = "./models/en_neuro_base-1.0/en_neuro_base/en_neuro_base-1.0"
SPACY_NORMAL_MODEL = "./models/en_tok2vec_binary_norm_v2-2.0.0/en_tok2vec_binary_norm_v2/en_tok2vec_binary_norm_v2-2.0.0"
SPACY_COMPARATIVE_MODEL = "./models/en_comp_cls-1.0/en_comp_cls/en_comp_cls-1.0"
SPACY_TOKENIZER = "./models/en_core_web_sm/en_core_web_sm-3.0.0"
SPACY_SECTIONER_PATH = "./models/en_tok2vec_section_cls-1.0/en_tok2vec_section_cls/en_tok2vec_section_cls-1.0"
SPACY_DOMAINER_PATH = "./models/pathology_patterns_v4"

DATA_FORMAT = srsly.read_json("./dashboard_assets/dashboard_config.json")


inference_models = {
    "comparative_cls": SPACY_COMPARATIVE_MODEL,
    "section_cls": SPACY_SECTIONER_PATH,
    "normality_cls": SPACY_NORMAL_MODEL,
    "info_model": SPACY_INFO_MODEL,
    "domainer": SPACY_DOMAINER_PATH,
    "tokenizer": SPACY_TOKENIZER,
}

st.set_page_config(layout="wide")
sidebar_title = "neuroNLP Dashboard"
sidebar_description = "A tool for interpreting neuroradiological reports"
inference_engine = model_factory(inference_models)
st.sidebar.title(sidebar_title)
st.sidebar.markdown(sidebar_description)
viewer = st.sidebar.selectbox(
    "Select view", ("Operational", "Clinical", "Workload Analysis")
)


ENCRYPTION_CHECKED = False
UPLOAD_COMPLETE = False
st.title("Operational Analysis Dashboard")
uploaded_files = st.file_uploader(
    "Upload CSV/XLSX", accept_multiple_files=True, type=["csv", "xlsx"]
)


filenames = get_filenames(uploaded_files)
uploaded_files = list(zip(filenames, uploaded_files))


@st.cache_data
def _get_report_df(list_of_files, data_format):
    report_dfs = read_file_input(list_of_files, data_format)
    report_df = pd.concat(report_dfs, ignore_index=True)
    inferred_cols = [key for key, vals in data_format.items() if vals["inferred"]]
    report_df = inference_engine.infer_addition_report_data(
        report_df, infer_data=inferred_cols, batch_size=128, n_processes=1
    )
    return report_df


@st.cache_data
def _prepare_data_for_workload(data):
    new_data = data.drop(columns=["pathological_domains"])
    new_data = transform_data(new_data)
    return new_data


if uploaded_files:
    encryption_list = check_file_encryption(uploaded_files)
    if any(encryption_list):
        password = st.text_input(
            "One or more files are password protected, please supply password"
        )
        if password:
            uploaded_files = decrypt_files(uploaded_files, encryption_list, password)
            ENCRYPTION_CHECKED = True
    else:
        ENCRYPTION_CHECKED = True


if ENCRYPTION_CHECKED:
    report_df = _get_report_df(uploaded_files, DATA_FORMAT)
    # workload_data = _prepare_data_for_workload(report_df)
    UPLOAD_COMPLETE = True


def select_dates(frame, min_date, max_date):
    selected_rows = frame[
        (frame["End Exam Date"].dt.date >= min_date)
        & (frame["End Exam Date"].dt.date < max_date)
    ]
    return selected_rows


if viewer == "Operational":
    if UPLOAD_COMPLETE:
        with st.expander("Report Selection"):
            report_subset, criteria_dict = selection_display(report_df, DATA_FORMAT)
        plot_to_save = plotting_display(report_subset, DATA_FORMAT)
        (
            continuous_summary_df,
            categorical_summary_df,
            basic_summary_df,
        ) = summary_display(report_subset, DATA_FORMAT)
        export_data = {
            "selection_criteria": criteria_dict,
            "plot": plot_to_save,
            "continuous_summary": continuous_summary_df,
            "categorical_summary": categorical_summary_df,
            "basic_summary": basic_summary_df,
        }
        run_name = "output_{}".format(str(date.today()))
        buffer = export_pdf(export_data, run_name)
        st.download_button(
            label="Export analysis as pdf", data=buffer, file_name="analysis_report.pdf"
        )
        csv = convert_df(report_subset)
        st.download_button(
            label="Export selection as csv",
            data=csv,
            file_name="data_selection.csv",
            mime="text/csv",
        )
        cat_csv = convert_df(categorical_summary_df)
        st.download_button(
            label="Export categorical variable summary",
            data=cat_csv,
            file_name="categorical_summary.csv",
            mime="text/csv",
        )
        cont_csv = convert_df(continuous_summary_df)
        st.download_button(
            label="Export continuous variable summary",
            data=cont_csv,
            file_name="continuous_summary.csv",
            mime="text/csv",
        )

elif viewer == "Clinical":
    if UPLOAD_COMPLETE:
        report_row = None
        with st.expander("Report Selection"):
            report_subset, criteria_dict = selection_display(report_df, DATA_FORMAT)
        report_row = report_search_display(report_subset)
        if report_row is not None:
            EXAMPLE_REPORT = report_row["Narrative"]
            if len(EXAMPLE_REPORT) > 5:
                doc = inference_engine.nlp(EXAMPLE_REPORT)
                clinical_display(report_row, doc)
            else:
                st.text("No report narrative present")

elif viewer == "Workload Analysis":
    st.header("Radiology workload analysis")
    if UPLOAD_COMPLETE:
        min_date, max_date = (
            report_df["End Exam Date"].min(),
            report_df["End Exam Date"].max(),
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            start_date = st.date_input(
                "Interval Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
            )
            end_date = st.date_input(
                "Interval End Date",
                value=max_date,
                min_value=start_date,
                max_value=max_date,
            )
        run_name = "output_{}".format(str(date.today()))
        run_dir = Path("./dashboard_assets/{}".format(run_name))
        if not run_dir.exists():
            os.system("mkdir {}".format(run_dir))
        transformed_data = _prepare_data_for_workload(report_df)
        selected_rows = select_dates(transformed_data, start_date, end_date)
        # transformed_data = aggregate_cerebrovasc(transformed_data)
        with col2:
            reporter_pdf = aggregate_reporters_report(selected_rows, run_dir)
            st.download_button(
                label="Export reporters analysis as pdf",
                data=reporter_pdf,
                file_name="total_reporter_analysis.pdf",
            )
            contrast_pdf = contrast_usage_report(selected_rows, run_dir)
            st.download_button(
                label="Export contrast usage analysis as pdf",
                data=contrast_pdf,
                file_name="contrast_usage_analysis.pdf",
            )
            for i, buf in enumerate(per_reporter_analysis(selected_rows, run_dir)):
                st.download_button(
                    label=f"Export analysis for reporter {i} as pdf",
                    data=buf,
                    file_name=f"reporter_{i}_analysis.pdf",
                )
