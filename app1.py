import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sklearn
import csv

if 'chats' not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = "Chat 1"
if 'first_run' not in st.session_state:
    st.session_state.first_run = {}

def clean_data(data):
    return data.dropna()  

def assess_column_types(data):
    column_types = {}
    for col in data.columns:
        unique_values = data[col].dropna().unique()
        dtype = data[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            if len(unique_values) == 2:
                column_types[col] = 'Binary'
            elif len(unique_values) <= 10:
                column_types[col] = 'Ordinal'
            else:
                column_types[col] = 'Ratio'
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            column_types[col] = 'Categorical'
        else:
            column_types[col] = 'Other'
    return column_types

def prepend_prompt_format(prompt, data, column_types):
    column_info = "\n".join([f"{col}: {col_type}" for col, col_type in column_types.items()])
    return (
        f"Your task is to give answer in two sections, First section will begin with #ANSWER# and following it would be one to two line answer. Second section will be code (if applicable) and it will begin with #CODE# followed by python code which will always be related to matplotlib visualization. Use 'st.pyplot()' instead of 'plt.show' as it is being displayed on streamlit. Import necessary libraries along with streamlit for the same. Also always assume that data is present in the 'data' variable. Don't modify data variable directly. Also remember that the code you generate will be given in the exec() function of python. Don't mention anything about code the user should not know that there is code. If code is not required at all or no visualization is asked, then create empty section #CODE#nocode."
        f"Data has been cleaned and standardized. Column types are as follows:\n{column_info}\n"
        f"Dataset: {data.head().to_string()}\nQuery: {prompt}"
    )

def show_user_message(message):
    st.chat_message("user").write(message['parts'][0])

def exec_chart_code(code, data):
    if not code:
        return None
    try:
        exec_locals = {}
        exec(code, {"plt": plt, "sns": sns, "pd": pd, 'sklearn': sklearn, "data": data}, exec_locals)
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        return plot_buffer
    except Exception as e:
        st.error(f"Error in executing the generated code: {str(e)}")
        return None

def show_assistant_message(message, data):
    answer = message.parts[0].text.split("#ANSWER#")[1].split("#CODE#")[0].strip()
    code = message.parts[0].text.split("#CODE#")[1].strip()
    if code.startswith('```python'):
        code = code[9:-3]
    if code.startswith('nocode'):
        code = ''
    st.chat_message("assistant").write(answer)
    if code:
        plot_buffer = exec_chart_code(code, data)
        if plot_buffer:
            st.image(plot_buffer)

genai.configure(api_key="API_KEY")
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

st.set_page_config(page_title="Data Vizard", layout="wide", page_icon="📊")

st.title("📊 Data Vizard")
st.sidebar.title("Chat Management")

chat_names = list(st.session_state.chats.keys())
selected_chat = st.sidebar.selectbox("Select Chat", chat_names + ["+ New Chat"])

if selected_chat == "+ New Chat":
    new_chat_name = st.sidebar.text_input("Enter a name for the new chat:")
    if st.sidebar.button("Create Chat"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            st.rerun()

else:
    st.session_state.current_chat = selected_chat

messages = st.session_state.chats[st.session_state.current_chat]
if st.session_state.current_chat not in st.session_state.first_run:
    st.session_state.first_run[st.session_state.current_chat] = False

st.markdown(" Upload Your Dataset Here")
uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
    sniffer = csv.Sniffer()
    detected_delimiter = sniffer.sniff(content.splitlines()[0]).delimiter
    uploaded_file.seek(0)  

    data = pd.read_csv(uploaded_file, delimiter=detected_delimiter, on_bad_lines="skip",)
    column_types = assess_column_types(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Here's a preview of your dataset:", data.head())
    st.write("Identified column types from your data:", column_types)
    st.write("### Column Descriptions")
    st.write(data.describe(include='all').T)

    st.write("### Chat Section")
    for message in messages:
        if hasattr(message, 'role') and message.role == 'model':
            show_assistant_message(message, data)
        elif message['role'] == 'user':
            show_user_message(message)

    prompt = st.chat_input("Enter your query")

    if prompt:
        messages.append({'role': "user", 'parts': [prompt]})
        show_user_message(messages[-1])

        conversation = []
        for message in messages:
            if hasattr(message, 'role') and message.role == 'model':
                conversation.append(message)
            elif message['role'] == 'user':
                conversation.append({
                    'role': 'user',
                    'parts': [prepend_prompt_format(message['parts'][0], data, column_types)]
                })

        response = model.generate_content(conversation)
        messages.append(response.candidates[0].content)
        show_assistant_message(messages[-1], data)
