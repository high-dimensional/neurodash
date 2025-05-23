"""Utilty functions for neuroDash"""

import base64
import io
import json
import os
from datetime import date
from pathlib import Path

import msoffcrypto
import pandas as pd
import streamlit as st
from neuradicon.custom_pipes import *
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from neurodash.inference import DashboardInferenceEngine


def process_ris_df(report_df, data_format):
    """Preprocess RIS-format CSV file into Pandas dataframe

    RIS CSV format from NHNN database, format from period 2018-10-28_to_2019-04-01
    """

    required_columns = [
        key for key, vals in data_format.items() if not vals["inferred"]
    ]

    if not all([name in report_df.columns for name in required_columns]):
        raise Exception(
            f"This CSV does not all have the required columns for RIS format. Required columns are {required_columns}"
        )
    report_df["Narrative"] = report_df["Narrative"].str.replace(
        r"\\\w+|\{.*?\}|}", "", regex=True
    )
    report_df["Narrative"] = report_df["Narrative"].str.replace(
        r"(\n+)", " ", regex=True
    )
    report_df["Narrative"] = report_df["Narrative"].str.replace(
        r"(\r+)", " ", regex=True
    )
    report_df["Narrative"] = report_df["Narrative"].str.replace(" .", ".", regex=False)
    report_df["Narrative"] = report_df["Narrative"].str.replace(
        r"_\S+_", "", regex=True
    )
    report_df["Narrative"] = report_df["Narrative"].str.replace(
        r"\s{2,}", " ", regex=True
    )
    report_df["Narrative"] = report_df["Narrative"].str.strip()
    return report_df


@st.cache_resource
def model_factory(inference_models):
    return DashboardInferenceEngine(inference_models)


def replace_normality_labels(df):
    df.loc[df["normality_class"] == "DEFACTO", "normality_class"] = (
        "NORMAL AS NO ASSERTION"
    )
    df.loc[df["normality_class"] == "STOCK", "normality_class"] = (
        "NORMAL USING STOCK PHRASE"
    )
    df.loc[df["normality_class"] == "INCIDENTAL", "normality_class"] = (
        "NORMAL WITH INCIDENTALS"
    )
    df.loc[df["normality_class"] == "CONTEXTUAL", "normality_class"] = (
        "NORMAL ACCORDING TO CONTEXT"
    )
    return df


@st.cache_resource(show_spinner=False)
def load_pipeline(name):
    # st.text("loading pipeline")
    return spacy.load(name)


def get_multiselect_options(pd_series, max_n_cats=30):
    """utility to get available options for multiselectbox"""
    options = pd_series.value_counts().index[:max_n_cats].tolist()
    options.append("all")
    try:
        options = sorted(options)
    except:
        pass
    return options


def get_multilabel_select_options(pd_series):
    options = (d for doc in pd_series.tolist() for d in doc)
    options = list(set(options))
    options.append("all")
    try:
        options = sorted(options)
    except:
        pass
    return options


def export_pdf(pdf_data, name):
    """create a pdf using reportlab and create download"""

    run_dir = Path("./dashboard_assets/{}".format(name))
    if not run_dir.exists():
        os.system("mkdir {}".format(run_dir))

    plot = pdf_data["plot"]
    selections = pdf_data["selection_criteria"]
    categorical_summary = pdf_data["categorical_summary"]
    continuous_summary = pdf_data["continuous_summary"]
    basic_summary = pdf_data["basic_summary"]
    FILENAME = "figure.png"
    plot.write_image(str(run_dir / FILENAME), format="png", scale=2)
    selection_table = json.loads(selections.to_json(orient="split", double_precision=3))
    categorical_summary_1 = categorical_summary.loc[:, categorical_summary.columns[:6]]
    categorical_summary_2 = categorical_summary.loc[
        :, categorical_summary.columns[6:12]
    ]
    categorical_summary_table_1 = json.loads(
        categorical_summary_1.to_json(orient="split", double_precision=3)
    )
    categorical_summary_table_2 = json.loads(
        categorical_summary_2.to_json(orient="split", double_precision=3)
    )
    continuous_summary_table = json.loads(
        continuous_summary.to_json(orient="split", double_precision=3)
    )
    # basic_summary_table = basic_summary.to_json(orient="split", double_precision=3)

    pdf_data = {
        "path": run_dir,
        "filename": name,
        "title_data": {"title": "Operational Analysis Report", "author": "USER"},
        "categorical_summary": categorical_summary_table_1,
        "categorical_summary_2": categorical_summary_table_2,
        "continuous_summary": continuous_summary_table,
        "image": str(run_dir / FILENAME),
        "variable_summary": selection_table,
    }
    file_buffer = generate_pdf(pdf_data)
    return file_buffer


EXAMPLE_TABLE = {
    "columns": ["col 1", "col 2"],
    "index": ["row 1", "row 2"],
    "data": [[235235, 23.2112314], ["c14134154", "d"]],
}
DATA = {
    "filename": "test",
    "title_data": {
        "title": "Operational Analysis Report",
        "author": "USER",
        "target": "operations",
    },
    "categorical_summary": EXAMPLE_TABLE,
    "continuous_summary": EXAMPLE_TABLE,
    "image": "checked.png",
    "variable_summary": EXAMPLE_TABLE,
}


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def make_title(title_data):
    title_style = ParagraphStyle(
        name="Title",
        alignment=TA_LEFT,
        fontSize=18,
        leading=15,
        font="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        alignment=TA_LEFT,
        fontSize=11,
        leading=15,
        font="Helvetica",
    )
    title = Paragraph(str(title_data["title"]), title_style)
    subtitle = Paragraph(
        f"Date: {str(date.today())}",
        subtitle_style,
    )
    table = Table([[title], [""], [subtitle]])
    return table


def make_service_report_title(title_data):
    title_style = ParagraphStyle(
        name="Title",
        alignment=TA_LEFT,
        fontSize=18,
        leading=15,
        font="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        alignment=TA_LEFT,
        fontSize=11,
        leading=15,
        font="Helvetica",
    )
    title = Paragraph(str(title_data["title"]), title_style)
    subtitle = Paragraph(
        f"Date: {str(date.today())}, Analysis for: {str(title_data['target'])}, Date range: {title_data['start date']} to {title_data['end date']}",
        subtitle_style,
    )
    table = Table([[title], [""], [subtitle]])
    return table


def make_table(table_data):
    """note: when using pandas.to_json, ensure it uses the split orientation"""
    data_style = ParagraphStyle(
        name="datum", alignment=TA_LEFT, fontSize=8, font="Helvetica"
    )
    columns = [""] + table_data["columns"]
    columns = [str(c) for c in columns]
    rows = [
        [name] + data for name, data in zip(table_data["index"], table_data["data"])
    ]
    reformatted_table_data = [columns] + rows
    table_style = TableStyle(
        [
            ("LINEBELOW", (0, 0), (len(columns), 0), 0.7, colors.black),
            ("ALIGNMENT", (0, 0), (-1, -1), "LEFT"),
        ]
    )
    table = Table(
        reformatted_table_data, style=table_style, spaceBefore=7, hAlign="LEFT"
    )
    return table


def make_image(image_filename, width=16, height=11):
    return Image(image_filename, width * cm, height * cm, hAlign="CENTER")


def df_to_json(df):
    return json.loads(df.to_json(orient="split", double_precision=3))


def get_filenames(file_list):
    return [f.name for f in file_list]


def generate_pdf(data):
    file_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        file_buffer,
        rightMargin=0.5 * cm,
        leftMargin=0.5 * cm,
        topMargin=0.5 * cm,
        bottomMargin=0.5 * cm,
    )

    title = make_title(data["title_data"])
    variable_summary = make_table(data["variable_summary"])
    categorical_summary = make_table(data["categorical_summary"])
    categorical_summary_2 = make_table(data["categorical_summary_2"])
    continuous_summary = make_table(data["continuous_summary"])
    image = make_image(data["image"])
    story = [
        title,
        variable_summary,
        categorical_summary,
        categorical_summary_2,
        continuous_summary,
        Spacer(0, 20),
        image,
    ]
    doc.build(story)
    return file_buffer


def generate_variable_pdf(data):
    HEIGHT = 9
    WIDTH = 15
    file_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        file_buffer,
        rightMargin=0.5 * cm,
        leftMargin=0.5 * cm,
        topMargin=0.5 * cm,
        bottomMargin=0.5 * cm,
    )

    title = make_title(data["title_data"])
    variable_summary = make_table(data["variable_summary"])
    image_1 = make_image(data["summary_image"], width=WIDTH, height=HEIGHT)
    story = [
        title,
        variable_summary,
        Spacer(0, 20),
        image_1,
    ]
    for key in data["breakdown_tables"].keys():
        story.append(
            make_breakdown(
                data["variable"],
                key,
                make_table(data["breakdown_tables"][key]),
                make_image(data["breakdown_images"][key], width=WIDTH, height=HEIGHT),
            )
        )
    doc.build(story)
    return file_buffer


def generate_service_report_pdf(data, HEIGHT=20, WIDTH=14):
    file_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        file_buffer,
        rightMargin=0.5 * cm,
        leftMargin=0.5 * cm,
        topMargin=0.5 * cm,
        bottomMargin=0.5 * cm,
    )

    title = make_service_report_title(data["title_data"])
    variable_summary = make_table(data["variable_summary"])
    image_1 = make_image(data["summary_image"], width=WIDTH, height=HEIGHT)
    story = [
        title,
        # Table([[image_1, variable_summary]])
        variable_summary,
        # Spacer(0, 20),
        image_1,
    ]
    doc.build(story)
    return file_buffer


def make_breakdown(variable, breakdown_variable, table, image):
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        alignment=TA_LEFT,
        fontSize=11,
        leading=15,
        font="Helvetica",
    )
    subtitle = Paragraph(
        f"{variable} vs. {breakdown_variable}",
        subtitle_style,
    )

    table = Table([[subtitle], [table], [image]])
    return table


def identify_filetype(filename):
    if filename[-3:] == "csv":
        return "csv"
    elif filename[-4:] == "xlsx":
        return "xlsx"
    else:
        raise Exception("Filetype not supported, supported filetypes: .xlsx, .csv")


def convert_df(df):
    return df.to_csv().encode("utf-8")


def check_file_encryption(file_list):
    encryption_list = []
    for name, f in file_list:
        file_type = identify_filetype(name)
        if file_type == "xlsx":
            if is_encrypted(f):
                encryption_list.append(True)
            else:
                encryption_list.append(False)
        else:
            encryption_list.append(False)

    return encryption_list


@st.cache_data()
def decrypt_files(file_list, encryption_status, password):
    output_file_tuples = []
    for file_tuple, encrypted in zip(file_list, encryption_status):
        filename, file = file_tuple
        if encrypted:
            decrypted_file = decrypt_xlsx(file, password)
            output_file_tuples.append((filename, decrypted_file))
        else:
            output_file_tuples.append((filename, file))
    return output_file_tuples


def read_file_input(file_list, data_format):
    report_dfs = []
    date_columns, data_types = derive_columns(data_format)
    for name, f in file_list:
        file_type = identify_filetype(name)
        if file_type == "xlsx":
            ris_df = pd.read_excel(
                f,
                parse_dates=date_columns,
                index_col=False,
                engine="openpyxl",
                dtype=data_types,
            )

        else:
            ris_df = pd.read_csv(
                f,
                low_memory=False,
                parse_dates=date_columns,
                dayfirst=True,
                index_col=False,
                dtype=data_types,
            )

        df = process_ris_df(ris_df, data_format)
        report_dfs.append(df)
    return report_dfs


def derive_columns(data_format):
    date_columns = [key for key, vals in data_format.items() if vals["dtype"] == "date"]
    str2type = {"string": str, "boolean": bool, "integer": int, "list": list}
    data_types = {
        key: str2type[vals["dtype"]]
        for key, vals in data_format.items()
        if (vals["dtype"] != "date") and (not vals["inferred"])
    }
    return date_columns, data_types


def decrypt_xlsx(file_bytes, password):
    office_file = msoffcrypto.OfficeFile(file_bytes)
    decrypted = io.BytesIO()
    office_file.load_key(password=password)
    office_file.decrypt(decrypted)
    return decrypted


def is_encrypted(file_bytes):
    try:
        office_file = msoffcrypto.OfficeFile(file_bytes)
        if office_file.is_encrypted():
            return True
        else:
            return False
    except:
        return False
