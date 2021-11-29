import streamlit as st
import os
import docx2txt
import pdfplumber
import utils
import extract_summary
import abstract_summary
import encoder_decoder

st.subheader("DocumentFiles")
docx_file = st.file_uploader("Upload Document", type=["txt"])
raw_text = ""
if docx_file is not None:
    flag = True
    if docx_file.type == "text/plain":
        # Read as string (decode bytes to string)
        raw_text = str(docx_file.read(), "utf-8")
        sample_text = raw_text[:500]
    else:
        st.error("Please upload txt file only !!")

activity1 = ["Extractive Summarization", "Abstractive Summarization"]
st.sidebar.header("Select Extractive Summarization Approach")

# create selectboxes
topic1 = "Extractive Summarization Approach"
ext_smry = ["", "Page Rank from Scratch", "Auto summarizer - summy"]

choice = st.sidebar.selectbox(
    "*******************************",
    ext_smry,
    format_func=lambda x: "Select an Option" if x == "" else x,
)
print("choice before ###", choice)
abst_smry = ["", "Encoder Decoder", "Pre trained T5 Transformer"]
st.sidebar.header("Extractive Summarization Approach")

choice2 = st.sidebar.selectbox(
    "*******************************",
    abst_smry,
    format_func=lambda x: "Select an Option" if x == "" else x,
)

selection = ""
if len(choice) > 1 and len(choice2) == 0:
    selection = choice
elif len(choice) == 0 and len(choice2) > 1:
    selection = choice2
elif len(choice) > 1 and len(choice2) > 1:
    st.error("Please select only one option")

if choice2 == "Encoder Decoder":
    if len(choice) == 0:
        st.write(
            "** Due to limited hardware resources , session got crashed and model could'nt be trained. Below is the Model summary of 3 stacked LSTM for the encoder **"
        )
        model_summary, summart_txt = encoder_decoder.stacked_lstm_encoder(
            max_story_len=2000
        )
        st.write(summart_txt)

if st.button("View a subset of your input Text"):
    if raw_text:
        sample_text = raw_text[:500]
        print("sample_text", sample_text)
        st.write(sample_text)

if selection:
    if raw_text and flag == True:
        input_text = utils.load_single_story(raw_text)
        input_text = input_text[0].get("story")
        if selection == "Auto summarizer - summy":
            st.write("** Summary **")
            summary = extract_summary.sumy_summarizer(input_text)
            st.write(summary)
        elif selection == "Page Rank from Scratch":
            st.write("** Summary **")
            summary = extract_summary.generate_summary(input_text)
            st.write(summary)
        elif selection == "Pre trained T5 Transformer":
            st.write("** Summary **")
            summary = abstract_summary.t5_summary(input_text)
            st.write(summary)
        elif selection == "Encoder Decoder":
            st.write(
                "** Due to limited hardware resources , session got crashed and model could'nt be trained. Below is the Model summary of 3 stacked LSTM for the encoder **"
            )
            model_summary, summart_txt = encoder_decoder.stacked_lstm_encoder(
                max_story_len=2000
            )
            st.write(summart_txt)
