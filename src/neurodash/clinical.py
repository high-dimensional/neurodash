""" Clinical display for neuroDash

Display and analysis tools for the clinical
view of the neuroDash dashboard
"""
import pandas as pd
import streamlit as st
from spacy import displacy
from spacy.tokens import Span

from neurodash.utils import get_html


def display_doc(doc):
    """Display the entities of a spacy doc"""
    options = {
        "colors": {
            "LOCATION-ASSERTED": "#5d99fd",
            "LOCATION-DENIED": "#ff1515",
            "PATHOLOGY-ASSERTED": "#5d99fd",
            "PATHOLOGY-DENIED": "#ff1515",
            "DESCRIPTOR-ASSERTED": "#5d99fd",
            "DESCRIPTOR-DENIED": "#ff1515",
        }
    }
    assertion_val_ents = [
        Span(doc, e.start, e.end, label=e.label_ + "-DENIED")
        if e._.is_negated
        else Span(doc, e.start, e.end, label=e.label_ + "-ASSERTED")
        for e in doc.ents
    ]
    doc.ents = assertion_val_ents
    html = displacy.render(doc, style="ent", options=options)
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


def report_search_display(df):
    st.header("Search")
    search_columns = [
        "MRN",
        "Dept Specialty",
        "Name",
        "End Exam Date",
    ]
    col1, col2 = st.columns([1, 2])
    with col1:
        patient_id = st.selectbox("Patient ID", sorted(df["MRN"].unique().tolist()))
        patient_rows = df[df["MRN"] == patient_id]
        if len(patient_rows) < 1:
            st.text("No reports for this date")
            return None
        selected_row_idx = st.selectbox(
            "Report row to select", patient_rows.index.values, index=0
        )
    with col2:
        st.dataframe(patient_rows[search_columns])

    selected_row = df.loc[selected_row_idx]
    return selected_row


def clinical_display(df_row, doc):
    st.header("Report")
    report_metadata_dict = {
        "Name": [str(df_row["Name"])],
        "Sex": [str(df_row["Sex"])],
        "Patient ID": [str(df_row["MRN"])],
        "Age": [str(df_row["Age"])],
        "Procedure": [str(df_row["Procedure"])],
        "With contrast?": [str(df_row["uses_contrast"])],
        "Normality class": [str(df_row["normality_class"])],
        "Compared to previous imaging?": [str(df_row["is_comparative"])],
        "Requesting Clinician": [str(df_row["Requesting Clinician"])],
        "Reporting Clinicians": [str(df_row["Reporting Clinicians"])],
        "Pathological domains": [", ".join(df_row["pathological_domains"])],
    }
    report_metadata_df = pd.DataFrame.from_dict(report_metadata_dict)

    entities_dict = [
        (e.text, ", ".join([l.text for l in e._.relation]), e._.is_negated)
        for e in doc.ents
        if e.label_ in ["PATHOLOGY", "DESCRIPTOR"]
    ]
    entities_df = pd.DataFrame.from_records(
        entities_dict, columns=["entity", "location", "assertion"]
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        st.table(report_metadata_df.transpose().rename(columns={0: "value"}))
        st.subheader("Asserted clinical entities")
        st.table(entities_df[~entities_df["assertion"]][["entity", "location"]])
        st.subheader("Denied clinical entities")
        st.table(entities_df[entities_df["assertion"]][["entity", "location"]])

    with col2:
        st.header("Clinical Concepts")
        display_doc(doc)
