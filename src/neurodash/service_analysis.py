"""Service analysis & workload utilities.

This module contains functions for automated service analysis details
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
from neurodash.utils import df_to_json, generate_service_report_pdf

sns.set_style("ticks")
FONTSIZE = 10
plt.rcParams["axes.labelsize"] = FONTSIZE
plt.rcParams["axes.titlesize"] = FONTSIZE
plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["legend.fontsize"] = FONTSIZE
plt.rcParams["legend.title_fontsize"] = FONTSIZE
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE

PATHOLOGICAL_DOMAINS = [
    "Interventional - Surgery",
    "Cerebrovascular",
    "Neoplastic & paraneoplastic",
    "Epilepsy",
    "Infectious",
    "Haematological",
    "Metabolic, Nutritional, & Toxic",
    "CSF disorders",
    "Ophthalmological",
    "Headache",
    "Endocrine",
    "Inflammatory & Autoimmune",
    "Neurodegenerative & Dementia",
    "Congenital & Developmental",
    "Traumatic",
    "Musculoskeletal",
]


def load_input(data_file):
    """load in data or take input from stdin"""
    df = pd.read_csv(data_file, low_memory=False, parse_dates=["End Exam Date"])
    return df


def add_pathology_columns(df):
    """replace pathology count columns with boolean columns"""
    pathology_columns = [c for c in df.columns if "asserted-pathology" in c]
    for c in pathology_columns:
        name = "has-" + c[19:]
        df.loc[:, name] = df.loc[:, c] > 0
    return df


def split_reporters(df):
    split = df["Reporting Clinicians"].str.split("\n", n=2, expand=True)
    df["primary_reporter"] = split[0]
    if len(split.columns) == 2:
        df["secondary_reporter"] = split[1]
    else:
        df["secondary_reporter"] = np.nan
    return df


def transform_data(data):
    """perform the necessary transformation on the input data"""

    transformed_data = (
        data.assign(
            Age=lambda x: pd.cut(
                x.Age,
                bins=[10 * i for i in range(11)],
                right=False,
                include_lowest=True,
                labels=[f"{10 * i}-{10*i+9}" for i in range(10)],
            )
        )
        .assign(date=lambda x: x["End Exam Date"].dt.date)
        .pipe(split_reporters)
    )
    return transformed_data


def create_pdf_data(var, table, image_path, start, end):
    DATA = {
        "filename": "test",
        "variable": var,
        "title_data": {
            "title": "Operational Analysis Report",
            "start date": str(start),
            "end date": str(end),
            "target": str(var),
        },
        "summary_image": image_path,
        "variable_summary": table,
    }
    return DATA


def output_results(name, results, plot, outdir, imheight, imwidth, start, end):
    """output analysis, produce a hard-formatted PDF output of plots and tables"""
    image_path = outdir / f"{name}_plot.png"
    plot.savefig(image_path, dpi=200)
    pdf_data = create_pdf_data(name, results, image_path, start, end)
    file_buffer = generate_service_report_pdf(pdf_data, HEIGHT=imheight, WIDTH=imwidth)
    # with open(outdir / f"{name}.pdf", "wb") as outfile:
    #    # Copy the BytesIO stream to the output file
    #    outfile.write(file_buffer.getbuffer())
    return file_buffer


def describe_total(df):
    """to replace the count row in describe() dfs"""
    total = df.sum(numeric_only=True).to_frame(name="total").T
    table = pd.concat([df.describe(), total], ignore_index=False).drop(index=["count"])
    return table


def contrast_describe(df):
    vcounts = df.value_counts()
    out_df = pd.DataFrame(
        data=[[vcounts[True]], [vcounts[False]], [vcounts.sum()]],
        columns=["uses_contrast"],
        index=["n_with_contrast", "n_without_contrast", "total"],
    )
    return out_df


@st.cache_data
def aggregate_reporters_report(df, output_dir):
    aggregate = (
        df.groupby("date").count()["MRN"].to_frame(name="n_daily_reports").reset_index()
    )
    start, end = df["date"].min(), df["date"].max()
    table = describe_total(aggregate)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=aggregate, x="date", y="n_daily_reports", ax=ax1).set_title(
        "Count of daily reports for all reporters"
    )
    # ax1.set_xticklabels(
    #    ax1.get_xticklabels(), rotation=60, horizontalalignment="right"
    # )
    ax1.tick_params(axis="x", rotation=60)
    ax1.grid(True)

    fig.tight_layout()
    buffer = output_results(
        "aggregate_data", df_to_json(table), fig, output_dir, 10, 15, start, end
    )
    plt.close()
    return buffer


@st.cache_data
def per_reporter_analysis(df, output_dir):
    ### INDIVIDUAL REPORTERS

    rows_to_duplicate = df.loc[~df["secondary_reporter"].isna()]
    start, end = df["date"].min(), df["date"].max()
    merged_reporters = pd.concat([df, rows_to_duplicate], ignore_index=True)
    merged_reporters = merged_reporters.assign(reporter=lambda x: x["primary_reporter"])
    duplicate_rows = merged_reporters.duplicated()
    merged_reporters.loc[duplicate_rows, "reporter"] = merged_reporters.loc[
        duplicate_rows, "secondary_reporter"
    ]
    data_to_plot = (
        merged_reporters.groupby(["date", "reporter"])
        .count()["MRN"]
        .to_frame(name="count")
        .assign(
            proportion_of_total=lambda y: y.groupby("date")["count"].transform(
                lambda x: x / x.sum()
            )
        )
    )
    all_reporters = merged_reporters["reporter"].value_counts().index
    buffers = []
    for r in all_reporters:
        name = r
        df = data_to_plot.xs(name, level="reporter")
        table = describe_total(df)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
        ax1_ = ax1.twinx()
        sns.lineplot(data=df, x="date", y="count", ax=ax1, color="blue", label="count")
        ax1.set_title("Report count and proportion of total reports")
        sns.lineplot(
            data=df,
            x="date",
            y="proportion_of_total",
            ax=ax1_,
            color="orange",
            label="proportion_of_total",
        )
        ax2.set_title("Number of reports of each pathological class")
        ax1.legend(loc="upper left")
        ax1.tick_params(axis="x", rotation=60)
        ax1.grid(True)
        ax1_.legend(loc="upper right")
        ax1_.grid(False)
        pathology_columns = [
            c for c in merged_reporters.columns if c in PATHOLOGICAL_DOMAINS
        ]
        pathology_counts = (
            merged_reporters.loc[
                merged_reporters["reporter"] == name, pathology_columns
            ]
            .astype(int)
            .sum(axis=0)
            .to_frame(name="count")
            .reset_index(names="pathology_class")
            .sort_values(by="count")
        )
        sns.barplot(data=pathology_counts, x="pathology_class", y="count", ax=ax2)
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=60, horizontalalignment="right"
        )
        ax2.grid(True)
        fig.tight_layout()
        buffer = output_results(
            name, df_to_json(table), fig, output_dir, 20, 14, start, end
        )
        plt.close()
        buffers.append(buffer)
    return buffers


def calculate_contrast_proportion(df):
    pathology_columns = [c for c in df.columns if c in PATHOLOGICAL_DOMAINS] + ["date"]
    with_contrast = df.loc[df["uses_contrast"], pathology_columns].set_index("date")
    no_contrast = df.loc[~df["uses_contrast"], pathology_columns].set_index("date")
    new_df = with_contrast / (with_contrast + no_contrast)
    new_df = new_df.fillna(0.0).reset_index()

    return new_df


def create_rolling_average(df):
    pathology_columns = [c for c in df.columns if c in PATHOLOGICAL_DOMAINS]
    for c in pathology_columns:
        df[c] = df[c].rolling(7).mean()
    return df


@st.cache_data
def contrast_usage_report(df, output_dir):
    pathology_columns = [c for c in df.columns if c in PATHOLOGICAL_DOMAINS]
    contrast_usage = (
        df[pathology_columns + ["date", "uses_contrast"]]
        .astype({i: int for i in pathology_columns})
        .groupby(["date", "uses_contrast"])
        .sum()
    ).reset_index()
    start, end = df["date"].min(), df["date"].max()
    contrast_proportion = calculate_contrast_proportion(contrast_usage)
    contrast_proportion = create_rolling_average(contrast_proportion)
    has_contrast = contrast_usage[contrast_usage["uses_contrast"]].drop(
        columns=["uses_contrast"]
    )
    has_contrast = create_rolling_average(has_contrast)

    has_contrast_plot = pd.melt(
        has_contrast, ["date"], value_name="count, weekly rolling average"
    )
    contrast_proportion_plot = pd.melt(
        contrast_proportion, ["date"], value_name="proportion, weekly rolling average"
    )
    table = contrast_describe(df[["uses_contrast"]])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))
    sns.lineplot(
        data=has_contrast_plot,
        x="date",
        y="count, weekly rolling average",
        ax=ax1,
        hue="variable",
        palette="tab20",
    )
    ax1.set_title("Contrast usage counts for each pathology class")
    ax1.tick_params(axis="x", rotation=60)
    ax1.legend(bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax1.grid(True)
    sns.lineplot(
        data=contrast_proportion_plot,
        x="date",
        y="proportion, weekly rolling average",
        ax=ax2,
        hue="variable",
        palette="tab20",
    )
    ax2.set_title("Contrast usage for each class, as a proportion of the class")
    ax2.tick_params(axis="x", rotation=60)
    ax2.legend(bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax2.grid(True)
    fig.tight_layout()
    buffer = output_results(
        "contrast_usage", df_to_json(table), fig, output_dir, 20, 16, start, end
    )
    plt.close()
    return buffer


def aggregate_cerebrovasc(df):
    df["has-cerebrovascular"] = df[
        ["has-cerebrovascular", "has-vascular", "has-ischaemic", "has-haemorrhagic"]
    ].any(axis="columns")
    df = df.drop(
        columns=[
            "has-metabolic-nutritional-toxic",
            "has-ischaemic",
            "has-vascular",
            "has-haemorrhagic",
            "has-musculoskeletal",
        ]
    )
    return df
