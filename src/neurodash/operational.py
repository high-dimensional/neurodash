"""Operational display for neuroDash

Display and analysis tools for the operational view
of the neuroDash dashboard
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from neurodash.utils import (get_multilabel_select_options,
                             get_multiselect_options)

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
PATHOLOGICAL_DOMAINS_WITH_ALL = [
    "all",
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


def selection_display(report_df, data_config):
    """display selection widgets to extract a subset of reports from a dataframe"""

    selection_variables = {
        key: vals for key, vals in data_config.items() if vals["in_selection_panel"]
    }
    panel_variables = [
        "End Exam Date",
        "Procedure",
        "Requesting Clinician",
        "Ordering Dept",
        "Dept Specialty",
        "Reporting Clinicians",
        "pathological_domains",
        "uses_contrast",
        "Quality Priority",
        "Base Pt Class",
        "normality_class",
        "Sex",
        "Age",
    ]
    if not all([i in selection_variables.keys() for i in panel_variables]):
        raise Exception(
            f"Selection variables do not contain necessary variables, necessary variables for panel are {panel_variables}, while contains {selection_variables.keys()}"
        )

    min_age, max_age = int(report_df["Age"].min()), int(report_df["Age"].max())
    min_date, max_date = (
        report_df["End Exam Date"].min(),
        report_df["End Exam Date"].max(),
    )

    select_vals = {}
    multilabel_select_vals = dict()
    multilabel_options = get_multilabel_select_options(
        report_df["pathological_domains"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        select_vals["Ordering Dept"] = st.multiselect(
            selection_variables["Ordering Dept"]["display_name"],
            get_multiselect_options(report_df["Ordering Dept"]),
            default="all",
        )
        select_vals["Requesting Clinician"] = st.multiselect(
            selection_variables["Requesting Clinician"]["display_name"],
            get_multiselect_options(report_df["Requesting Clinician"]),
            default="all",
        )
        select_vals["Procedure"] = st.multiselect(
            selection_variables["Procedure"]["display_name"],
            get_multiselect_options(report_df["Procedure"]),
            default="all",
        )
        select_vals["Reporting Clinicians"] = st.multiselect(
            selection_variables["Reporting Clinicians"]["display_name"],
            get_multiselect_options(report_df["Reporting Clinicians"]),
            default="all",
        )
        multilabel_select_vals["pathological_domains"] = st.multiselect(
            selection_variables["pathological_domains"]["display_name"],
            multilabel_options,
            default="all",
        )

    with col2:
        select_vals["Dept Specialty"] = st.multiselect(
            selection_variables["Dept Specialty"]["display_name"],
            get_multiselect_options(report_df["Dept Specialty"]),
            default="all",
        )
        select_vals["uses_contrast"] = st.multiselect(
            selection_variables["uses_contrast"]["display_name"],
            get_multiselect_options(report_df["uses_contrast"]),
            default="all",
        )
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

    with col3:
        select_vals["Base Pt Class"] = st.multiselect(
            selection_variables["Base Pt Class"]["display_name"],
            get_multiselect_options(report_df["Base Pt Class"]),
            default="all",
        )

        select_vals["Quality Priority"] = st.multiselect(
            selection_variables["Quality Priority"]["display_name"],
            get_multiselect_options(report_df["Quality Priority"]),
            default="all",
        )

        select_vals["normality_class"] = st.multiselect(
            selection_variables["normality_class"]["display_name"],
            get_multiselect_options(report_df["normality_class"]),
            default="all",
        )

        min_select_age, max_select_age = st.slider(
            selection_variables["Age"]["display_name"],
            min_age,
            max_age,
            [min_age, max_age],
        )
        select_vals["Sex"] = st.multiselect(
            selection_variables["Sex"]["display_name"],
            get_multiselect_options(report_df["Sex"]),
            default="all",
        )

    selection_values_fill_all = {
        (key if "all" in val else key): (
            report_df[key].unique().tolist() if "all" in val else val
        )
        for key, val in select_vals.items()
    }

    if "all" in multilabel_select_vals["pathological_domains"]:
        domain_criterion = True
    else:
        domain_criterion = report_df[
            multilabel_select_vals["pathological_domains"]
        ].all(axis="columns")

    selection_criteria = (
        (report_df["Age"].between(min_select_age, max_select_age))
        & (report_df["End Exam Date"].dt.date.between(start_date, end_date))
        & (report_df["Sex"].isin(selection_values_fill_all["Sex"]))
        & (report_df["uses_contrast"].isin(selection_values_fill_all["uses_contrast"]))
        & (report_df["Procedure"].isin(selection_values_fill_all["Procedure"]))
        & (report_df["Ordering Dept"].isin(selection_values_fill_all["Ordering Dept"]))
        & (
            report_df["Dept Specialty"].isin(
                selection_values_fill_all["Dept Specialty"]
            )
        )
        & (
            report_df["Requesting Clinician"].isin(
                selection_values_fill_all["Requesting Clinician"]
            )
        )
        & (
            report_df["Reporting Clinicians"].isin(
                selection_values_fill_all["Reporting Clinicians"]
            )
        )
        & (
            report_df["Quality Priority"].isin(
                selection_values_fill_all["Quality Priority"]
            )
        )
        & (
            report_df["normality_class"].isin(
                selection_values_fill_all["normality_class"]
            )
        )
        & domain_criterion
    )

    selection_output = {}
    selection_output["min age"] = [str(min_select_age)]
    selection_output["max age"] = [str(max_select_age)]
    selection_output["start date"] = [str(start_date)]
    selection_output["end date"] = [str(end_date)]
    selection_table = pd.DataFrame.from_dict(selection_output)
    report_subset = report_df.loc[selection_criteria]

    return report_subset, selection_table


ALLVIEWS = [
    "Integrated",
    "Integrated - logarithmic",
    "Temporal",
    "Temporal - % change",
    "Temporal - proportion",
]


def plot_categorical(
    df,
    column_name,
    variable_name,
    logview=False,
):
    counts = (
        df[column_name]
        .value_counts(normalize=False)
        .rename_axis(variable_name)
        .to_frame("counts")
        .reset_index()
    )
    fig = px.bar(counts, x=variable_name, y="counts", log_y=logview)
    return fig


def plot_categorical_temporal(
    df,
    column_name,
    variable_name,
    temporal_variable="End Exam Date",
    period="M",
    logview=False,
    percentchange=False,
    proportion=False,
):
    classes = df[[temporal_variable, column_name]]
    # classes[column_name] = classes[column_name].fillna("Undefined")
    interval_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "A",
    }

    if percentchange:
        interval_name = st.selectbox("Interval", list(interval_map.keys()), index=2)
        interval = interval_map[interval_name]
        classes = classes.assign(counts=1)
        table = (
            pd.pivot_table(
                classes,
                values="counts",
                index=[temporal_variable],
                columns=[column_name],
                aggfunc="sum",
            )
            .resample(interval)
            .sum()
        )
        pc_table = table.pct_change() * 100
        fig = px.line(pc_table, x=pc_table.index, y=pc_table.columns, log_y=logview)
    elif proportion:
        nbins = st.slider("Number of interval bins:", 1, 100, 50)
        fig = px.histogram(
            classes,
            x=temporal_variable,
            color=column_name,
            log_y=logview,
            barnorm="percent",
            nbins=nbins,
        )
    else:
        nbins = st.slider("Number of interval bins:", 1, 100, 50)
        fig = px.histogram(
            classes, x=temporal_variable, color=column_name, log_y=logview, nbins=nbins
        )
    return fig


def plot_continuous(df, column_name, logview=False):
    nbins = st.slider("Number of histogram bins:", 1, 100, 50)
    fig = px.histogram(df, x=column_name, log_y=logview, nbins=nbins)
    return fig


def plot_continuous_temporal(
    df,
    column_name,
    temporal_variable="End Exam Date",
    logview=False,
    percentchange=False,
):
    nbinsy = st.slider("Number of heatmap bins:", 1, 100, 50)
    nbinsx = st.slider("Number of interval bins:", 1, 100, 50)
    fig = px.density_heatmap(
        df, x=temporal_variable, y=column_name, nbinsx=nbinsx, nbinsy=nbinsy
    )
    return fig


def plotter(report_df, plot_type, column_name, plot_view):
    if plot_type == "categorical":
        if plot_view == "Temporal":
            c = plot_categorical_temporal(report_df, column_name, column_name)
        elif plot_view == "Temporal - % change":
            c = plot_categorical_temporal(
                report_df, column_name, column_name, percentchange=True
            )
        elif plot_view == "Temporal - proportion":
            c = plot_categorical_temporal(
                report_df, column_name, column_name, proportion=True
            )
        elif plot_view == "Integrated - logarithmic":
            c = plot_categorical(report_df, column_name, column_name, logview=True)
        else:
            c = plot_categorical(report_df, column_name, column_name)

    elif plot_type == "continuous":
        if plot_view == "Temporal":
            c = plot_continuous_temporal(report_df, column_name)
        elif plot_view == "Temporal - % change":
            c = plot_continuous_temporal(report_df, column_name, percentchange=True)
        elif plot_view == "Integrated - logarithmic":
            c = plot_continuous(report_df, column_name, logview=True)
        else:
            c = plot_continuous(report_df, column_name)

    elif plot_type == "per_patient_categorical":
        unique_patient_df = report_df[~report_df["MRN"].duplicated(keep="first")]
        if plot_view == "Integrated - logarithmic":
            c = plot_categorical(
                unique_patient_df, column_name, column_name, logview=True
            )
        else:
            c = plot_categorical(unique_patient_df, column_name, column_name)

    elif plot_type == "per_patient_continuous":
        per_patient_df = (
            report_df["MRN"]
            .value_counts(normalize=False)
            .to_frame("n_scans")
            .reset_index()
        )
        if plot_view == "Integrated - logarithmic":
            c = plot_continuous(per_patient_df, "n_scans", logview=True)
        else:
            c = plot_continuous(per_patient_df, "n_scans")

    elif plot_type == "temporal":
        temporal_count_df = report_df
        temporal_count_df.loc[:, "report_count"] = "N"
        if plot_view == "Temporal":
            c = plot_categorical_temporal(
                temporal_count_df, "report_count", "report_count"
            )
        elif plot_view == "Temporal - % change":
            c = plot_categorical_temporal(
                temporal_count_df, "report_count", "report_count", percentchange=True
            )
        elif plot_view == "Temporal - logarithmic":
            c = plot_categorical_temporal(
                temporal_count_df, "report_count", "report_count", logview=True
            )
        else:
            c = plot_categorical_temporal(
                temporal_count_df, "report_count", "report_count"
            )

    st.plotly_chart(c, use_container_width=True)
    return c


def plotting_display(df, data_config):
    report_df = df
    st.header("Plotting")
    col1, col2 = st.columns(2)
    plottable_variables = {
        vals["display_name"]: key
        for key, vals in data_config.items()
        if vals["plot_type"] != "not_plottable"
    }
    with col1:
        variable_selection = st.selectbox(
            "Variable to plot", list(plottable_variables.keys())
        )
    with col2:
        to_plot = plottable_variables[variable_selection]
        VIEWS = data_config[to_plot]["allowed_plot_views"]
        plot_view = st.selectbox("Plot view", VIEWS)
    plot_type = data_config[to_plot]["plot_type"]
    c = plotter(report_df, plot_type, to_plot, plot_view)
    return c


def summary_display(report_df, data_config):
    st.header("Summary")
    n_reports = len(report_df)
    min_date, max_date = (
        report_df["End Exam Date"].dt.date.min(),
        report_df["End Exam Date"].dt.date.max(),
    )
    n_unique_patients = len(report_df["MRN"].unique())
    basic_variables = {
        "start_date": [min_date],
        "end_date": [max_date],
        "n_reports": [n_reports],
        "n_unique_patients": [n_unique_patients],
    }
    basic_summary_description = pd.DataFrame.from_dict(basic_variables)

    continuous_variables = [
        key
        for key, vals in data_config.items()
        if vals["plot_type"] in ["continuous", "per_patient_continuous"]
    ]
    continuous_summary_description = report_df[continuous_variables].describe()

    categorical_variables = [
        key
        for key, vals in data_config.items()
        if vals["plot_type"] in ["categorical", "per_patient_categorical"]
    ]
    categorical_summary_description = report_df[categorical_variables].describe()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.table(basic_summary_description)
    with col2:
        st.table(continuous_summary_description)
    st.table(categorical_summary_description.astype(str))
    return (
        continuous_summary_description,
        categorical_summary_description,
        basic_summary_description,
    )
