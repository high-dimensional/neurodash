#!/usr/bin/env python
"""Analysis of service workload.

This script runs analysis of radiological service usage
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import toml
from neuroNLP.dashboard.utils import df_to_json, generate_variable_pdf

sns.set()
FONTSIZE = 10
plt.rcParams["axes.labelsize"] = FONTSIZE
plt.rcParams["axes.titlesize"] = FONTSIZE
plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["legend.fontsize"] = FONTSIZE
plt.rcParams["legend.title_fontsize"] = FONTSIZE
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE


def load_input(data_file, config_file):
    """load in data or take input from stdin"""
    df = pd.read_csv(data_file, low_memory=False, parse_dates=["End Exam Date"])
    with open(config_file, "r") as file:
        config = toml.load(file)
    return df, config


def aggregate_categories(df, columns, top_k=10):
    """replace low-frequency classes with 'other' - top k retained"""
    for c in columns:
        top_ten = df[c].value_counts()[:top_k].index
        df.loc[~df[c].isin(top_ten), c] = "other"
    return df


def add_pathology_columns(df):
    """replace pathology count columns with boolean columns"""
    pathology_columns = [c for c in df.columns if "asserted-pathology" in c]
    for c in pathology_columns:
        name = "has-" + c[19:]
        df.loc[:, name] = df.loc[:, c] > 0
    return df


def transform_data(data, config):
    """perform the necessary transformation on the input data"""
    start = pd.to_datetime(
        config["dates"]["start"], dayfirst=True, infer_datetime_format=True
    )
    end = pd.to_datetime(
        config["dates"]["end"], dayfirst=True, infer_datetime_format=True
    )
    categorical_variables = [
        i["name"] for _, i in config["to_plot"].items() if i["name"] != "age_at_study"
    ]
    transformed_data = (
        data[(data["End Exam Date"] >= start) & (data["End Exam Date"] < end)]
        .assign(
            age_at_study=lambda x: pd.cut(
                x.age_at_study,
                bins=[10 * i for i in range(11)],
                right=False,
                include_lowest=True,
                labels=[f"{10 * i}-{10*i+9}" for i in range(10)],
            )
        )
        .assign(
            date=lambda x: (x["End Exam Date"] - pd.Timestamp("1970-01-01"))
            // pd.Timedelta("1s")
        )
        .pipe(add_pathology_columns)
        .fillna(value={j: "UNK" for j in categorical_variables})
        .astype({i: str for i in categorical_variables})
        .pipe(aggregate_categories, categorical_variables)
    )
    return transformed_data


def inference(data):
    """apply ml-algorithms to infer new variables"""
    return data


def analysis(data, config):
    """perform analysis on data"""
    analysis_outputs = {}
    for _, var_dict in config["to_plot"].items():
        summary_table = df_to_json(
            data[[var_dict["name"]] + var_dict["breakdown"]].describe()
        )
        breakdown_tables = dict()
        for c in var_dict["breakdown"]:
            breakdown_tables[c] = df_to_json(
                pd.crosstab(data[var_dict["name"]], data[c])
            )
        analysis_outputs[var_dict["name"]] = {
            "summary": summary_table,
            "breakdown_tables": breakdown_tables,
        }
    return analysis_outputs


def plot_result(data_to_plot, config):
    """plot the results"""
    plots = {}
    start, end = (
        data_to_plot["End Exam Date"].min(),
        data_to_plot["End Exam Date"].max(),
    )
    bins = pd.date_range(start=start, end=end, freq="QS")
    intbins = (bins - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    for _, var_dict in config["to_plot"].items():
        fig1 = plot_summary(data_to_plot, var_dict, bins, intbins)
        breakdown_plots = dict()
        for i, bd in enumerate(var_dict["breakdown"]):
            breakdown_plots[bd] = plot_breakdown(data_to_plot, var_dict, bd)
        plots[var_dict["name"]] = {"summary": fig1, "breakdowns": breakdown_plots}
    return plots


def plot_summary(data, var_dict, date_bins, int_date_bins):
    fig1, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=data,
        x="date",
        multiple="stack",
        hue=var_dict["name"],
        bins=int_date_bins,
        stat=var_dict["type"],
        ax=ax,
        palette="tab10",
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xticks(int_date_bins, labels=[b.strftime("%Y-%m-%d") for b in date_bins])
    plt.xticks(rotation=45)
    fig1.tight_layout()
    return fig1


def plot_breakdown(data, var_dict, breakdown_var):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # sns.countplot(
    #    data=data_to_plot, x=var_dict["name"], hue=bd, palette="tab10", ax=ax2
    # )
    sns.histplot(
        data=data,
        x=var_dict["name"],
        hue=breakdown_var,
        stat=var_dict["type"],
        palette="tab10",
        ax=ax2,
        multiple="dodge",
    )
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    fig2.tight_layout()
    return fig2


def create_pdf_data(var, tables, image_paths):
    DATA = {
        "filename": "test",
        "variable": var,
        "title_data": {"title": "Operational Analysis Report", "author": "USER"},
        "summary_image": image_paths["summary_path"],
        "variable_summary": tables["summary"],
        "breakdown_images": image_paths["breakdowns"],
        "breakdown_tables": tables["breakdown_tables"],
    }
    return DATA


def output_results(results, plots, outdir):
    """output analysis, produce a hard-formatted PDF output of plots and tables"""
    for var, plot_dict in plots.items():
        path_dict = dict()
        image_path = outdir / f"{var}_temporal.png"
        temporal_plot = plot_dict["summary"]
        temporal_plot.savefig(image_path, dpi=200)
        path_dict["summary_path"] = image_path
        path_dict["breakdowns"] = {}
        for b_var, b_plot in plot_dict["breakdowns"].items():
            breakdown_image_path = outdir / f"{var}_breakdown_{b_var}.png"
            b_plot.savefig(breakdown_image_path, dpi=200)
            path_dict["breakdowns"][b_var] = breakdown_image_path
        pdf_data = create_pdf_data(var, results[var], path_dict)
        file_buffer = generate_variable_pdf(pdf_data)
        with open(outdir / f"{var}.pdf", "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(file_buffer.getbuffer())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="report CSV", type=Path)
    parser.add_argument("config", help="analysis config file", type=Path)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()
    print("loading data")
    data, config = load_input(args.input, args.config)
    transformed_data = transform_data(data, config)
    print("running analysis")
    results = analysis(transformed_data, config)
    print("creating plots")
    plots = plot_result(transformed_data, config)
    print(f"output results to {args.outdir}")
    output_results(results, plots, args.outdir)


if __name__ == "__main__":
    main()
