

import streamlit as st
import requests
import pandas as pd
import openai
import os
from docx import Document
import re

# Paths
TRANSCRIPTIONS_DIR = "client_details/transcriptions"
CHECKS_FILE = "static/checks.csv"
QUERY_INTENTS_FILE = "static/query_intents.csv"

# Load files
checks_df = pd.read_csv(CHECKS_FILE)
query_intents_df = pd.read_csv(QUERY_INTENTS_FILE)


# Initialize OpenAI client
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])

# Fixed intent list for IFA Query
VALID_INTENTS = [
    "lookup client data",
    "inheritance planning",
    "non-inheritance tax optimisation",
    "retirement planning",
    "unemployment planning"
]

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "evaluate_transcription_active" not in st.session_state:
    st.session_state.evaluate_transcription_active = False
if "ifa_query_active" not in st.session_state:
    st.session_state.ifa_query_active = False

# Custom CSS to enhance Running icon
st.markdown("""
    <style>
    div[data-testid="stStatusWidget"] svg {
        width: 30px !important;
        height: 30px !important;
        color: #FF4500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# UI setup
st.title("Evaluation Tool")

# Workflow selection buttons
st.subheader("Choose a Workflow")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Initial Client Meeting"):
        st.session_state.workflow = "initial_client_meeting"
with col2:
    if st.button("IFA Query"):
        st.session_state.workflow = "ifa_query"
        st.session_state.ifa_query_active = True
with col3:
    if st.button("Initial Meeting Prep (n/a)"):
        st.session_state.workflow = "initial_meeting_prep"

# Workflow logic
if st.session_state.workflow == "initial_meeting_prep":
    st.write("Workflow: Initial Meeting Prep")
    st.write("n/a - No functionality implemented yet.")

elif st.session_state.workflow == "initial_client_meeting":
    st.write("Workflow: Initial Client Meeting")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Evaluate Transcription"):
            st.session_state.evaluate_transcription_active = True
    with col2:
        transcribe_audio = st.button("Transcribe Audio (n/a)")

    if transcribe_audio:
        st.write("n/a - Transcription functionality not implemented yet.")

    if st.session_state.evaluate_transcription_active:
        if not os.path.exists(TRANSCRIPTIONS_DIR):
            st.error(f"Transcriptions directory '{TRANSCRIPTIONS_DIR}' not found.")
        else:
            transcription_files = [f for f in os.listdir(TRANSCRIPTIONS_DIR) if f.endswith(".docx")]
            if not transcription_files:
                st.warning("No .docx files found in the transcriptions directory.")
            else:
                selected_file = st.selectbox("Choose a transcription file", transcription_files)
                
                file_path = os.path.join(TRANSCRIPTIONS_DIR, selected_file)
                doc = Document(file_path)
                transcription = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

                if not transcription:
                    st.warning("The selected file is empty.")
                else:
                    st.write("Transcription Preview:")
                    st.text_area("Content", transcription, height=200, disabled=True)

                    default_instruction = (
                        "The following is a transcription between a financial advisor and a client. Extract the below information from it, "
                        "and if the information isn't covered in the conversation indicate with n/a.\n\n"
                        "IFA status\nServices provided\nFees\nComplaints process\nAge\nMarital status\nChildren\n"
                        "AML ID check\nGoals\nInvesting experience\nIncome\nAssets\nLiabilities\nExpenses\nHealth\n"
                        "Insurance\nRisk tolerance\nInheritance plans\nISA utilisation"
                    )
                    prompt_instruction = st.text_area("Edit the prompt sent to the LLM", default_instruction, height=250)

                    st.write("Choose an evaluation model:")
                    col1, col2 = st.columns(2)
                    with col1:
                        eval_gpt4o_mini = st.button("Evaluate with GPT-4o Mini (cheapest)")
                    with col2:
                        eval_gpt35_turbo = st.button("Evaluate with GPT-3.5-turbo")

                    if (eval_gpt4o_mini or eval_gpt35_turbo) and prompt_instruction:
                        status = st.warning("Processing... Please wait.", icon="⚙️")
                        model = "gpt-4o-mini" if eval_gpt4o_mini else "gpt-3.5-turbo"
                        full_prompt = f"{prompt_instruction}\n\n{transcription}"
                        
                        try:
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant evaluating meeting transcriptions."},
                                    {"role": "user", "content": full_prompt}
                                ],
                                max_tokens=500
                            )
                            
                            status.empty()
                            eval_result = response.choices[0].message.content
                            st.write(f"Evaluation Result (using {model}):")
                            st.write(eval_result)

                            # Parse API response
                            response_lines = eval_result.split("\n")
                            response_dict = {}
                            for line in response_lines:
                                match = re.match(r"^-?\s*\**(.+?)\**:\s*(.*)$", line.strip())
                                if match:
                                    key, value = match.groups()
                                    key = key.replace("*", "").strip().lower()
                                    response_dict[key] = value.strip()

                            # Validation and scoring
                            client_name = " ".join(selected_file.replace(".docx", "").split("_")).title()
                            if client_name not in checks_df.columns[2:]:
                                st.error(f"Client '{client_name}' not found in checks.csv.")
                            else:
                                # Build table data
                                table_data = []
                                for index, row in checks_df.iterrows():
                                    item = row["Which"].lower().strip()
                                    actual = "Yes" if row[client_name] == "Yes" else "No"
                                    detected = response_dict.get(item, "n/a")
                                    table_data.append({
                                        "Type": row["Type"],
                                        "Which": row["Which"],
                                        "Actual": actual,
                                        "Detected": detected
                                    })

                                # Calculate category scores
                                st.write("Category Scores:")
                                categories = checks_df["Type"].unique()
                                for category in categories:
                                    category_items = checks_df[checks_df["Type"] == category]
                                    total = len(category_items)
                                    score = 0
                                    for index, row in category_items.iterrows():
                                        item = row["Which"].lower().strip()
                                        actual = row[client_name] == "Yes"
                                        detected = response_dict.get(item, "n/a") != "n/a"
                                        if actual == detected:  # Match if both True (present) or both False (n/a)
                                            score += 1
                                    st.write(f"{category}: {score}/{total}")

                                # Display validation table
                                st.write("Validation Table:")
                                validation_df = pd.DataFrame(table_data)
                                st.dataframe(validation_df)

                        except Exception as e:
                            status.empty()
                            st.write(f"Error calling OpenAI API: {e}")

elif st.session_state.workflow == "ifa_query" and st.session_state.ifa_query_active:
    st.write("Workflow: IFA Query")
    
    selected_query = st.selectbox("Select a query", query_intents_df["Query"].tolist())
    
    if selected_query:
        expected_intents = [intent.lower() for intent in query_intents_df[query_intents_df["Query"] == selected_query]["Intent"].iloc[0].split(", ")]
        
        default_prompt = (
            "Analyse this query and identify what tasks the user wanted to achieve, out of this list: "
            "Lookup client data, Inheritance planning, Non-inheritance tax optimisation, Retirement planning, Unemployment planning\n\n"
            f"Query: {selected_query}"
        )
        prompt = st.text_area("Edit the prompt sent to the LLM", default_prompt, height=200)
        
        st.write("Choose an evaluation model:")
        col1, col2 = st.columns(2)
        with col1:
            eval_gpt4o_mini = st.button("Evaluate with GPT-4o Mini (cheapest)")
        with col2:
            eval_gpt35_turbo = st.button("Evaluate with GPT-3.5-turbo")

        if (eval_gpt4o_mini or eval_gpt35_turbo) and prompt:
            status = st.warning("Processing... Please wait.", icon="⚙️")
            model = "gpt-4o-mini" if eval_gpt4o_mini else "gpt-3.5-turbo"
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an assistant that identifies intents from financial advisor queries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200
                )
                
                status.empty()
                llm_result = response.choices[0].message.content
                st.write(f"LLM Analysis Result (using {model}):")
                st.write(llm_result)

                detected_intents = []
                for intent in VALID_INTENTS:
                    if intent.lower() in llm_result.lower():
                        detected_intents.append(intent.lower())

                st.write("Intent Validation:")
                total_intents = len(expected_intents)
                correct_intents = 0
                discrepancies = []

                for intent in expected_intents:
                    if intent in detected_intents:
                        st.write(f"- {intent}: Correctly identified")
                        correct_intents += 1
                    else:
                        st.write(f"- {intent}: Missed by LLM")
                        discrepancies.append(f"Missed: {intent}")

                for intent in detected_intents:
                    if intent not in expected_intents:
                        st.write(f"- {intent}: Incorrectly identified")
                        discrepancies.append(f"Extra: {intent}")

                score = f"{correct_intents}/{total_intents}"
                if discrepancies:
                    st.warning(f"Score: {score}")
                    st.write("Discrepancies:")
                    for d in discrepancies:
                        st.write(f"- {d}")
                    st.write("Full LLM Response (for debugging):")
                    st.write(llm_result)
                else:
                    st.success(f"Score: {score} - Perfect match!")

            except Exception as e:
                status.empty()
                st.write(f"Error calling OpenAI API: {e}")