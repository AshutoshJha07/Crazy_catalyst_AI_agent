import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import requests
from io import StringIO

st.set_page_config(
    page_title="CRAZY CATALYST",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CRAZY CATALYST - Advanced Business Analyst")
st.markdown(
    "**Welcome to CRAZY CATALYST!** "
    "Upload your business data and have a conversation with an AI agent that can interpret, "
    "analyze trends, and help craft targeted marketing campaigns."
)

# ---------------------
#     SIDEBAR
# ---------------------
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

st.sidebar.subheader("Advanced Settings")
temperature = st.sidebar.slider("Temperature (creativity of responses)", 0.0, 1.0, 0.6, 0.1)
max_tokens = st.sidebar.slider("Max tokens (length of AI response)", 50, 1500, 300, 50)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head(10))

    st.write("### Summary Statistics")
    st.write(df.describe())

    with st.expander("Show Correlation Heatmap"):
        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.shape[1] > 1:
            corr_matrix = numeric_cols.corr()
            fig_corr = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale='Blues',
                showscale=True
            )
            st.plotly_chart(fig_corr)
        else:
            st.warning("Not enough numeric columns to create a correlation heatmap.")

    if "Revenue" in df.columns and "Product" in df.columns:
        with st.expander("Show Revenue by Product Chart"):
            fig = px.bar(df, x="Product", y="Revenue", title="Revenue by Product")
            st.plotly_chart(fig)

st.markdown("---")
st.subheader("Chat with CRAZY CATALYST")

if df is None:
    st.info("Please upload a CSV file in the sidebar to start chatting with CRAZY CATALYST.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the current conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about your data (e.g., 'What are key trends in revenue?')")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # SYSTEM PROMPT
        # Limit columns to first 5 only
        columns_to_show = df.columns[:5]
        # Round summary stats to 2 decimal places
        df_summary = df.describe().round(2).to_string()

        system_prompt = f"""
        You are CRAZY CATALYST, a highly advanced AI assistant specializing in business analysis 
        and data insights. You can interpret data, identify trends, and provide targeted marketing 
        recommendations based on business metrics.

        Important Guidelines:
        1. Do not reveal or describe your chain-of-thought, internal reasoning, or any hidden analysis.
        2. Provide clear, concise, and actionable insights.
        3. Focus on responding directly to the user’s queries without unnecessary elaboration.
        4. Maintain a professional yet helpful tone.

        When you respond, provide only your final answer and avoid disclosing any hidden steps 
        or reasoning.

        -- DATA CONTEXT --
        (Showing only the first 5 columns to conserve space)
        Columns: {', '.join(columns_to_show)}
        
        Summary (Rounded to 2 decimals):
        {df_summary}
        -- END DATA CONTEXT --
        """

        # Keep the last N messages to manage context length
        N = 6
        truncated_history = st.session_state.messages[-N:]

        model_messages = [{"role": "system", "content": system_prompt}]
        model_messages.extend(truncated_history)

        MODEL_ENDPOINT = "http://localhost:1234/v1/chat/completions"

        payload = {
            "model": "deepseek-r1-qwen-7b",  # Replace if your local model name differs
            "messages": model_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                MODEL_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            assistant_reply = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as http_err:
            # If the server returns a 400 error, show a simple non-creative message
            if http_err.response.status_code == 400:
                assistant_reply = (
                    "Error 400: The request was invalid or too large. "
                    "Try reducing chat history or data context, or check server logs."
                )
            else:
                assistant_reply = f"Error calling the model: {str(http_err)}"
        except Exception as e:
            assistant_reply = f"Error calling the model: {str(e)}"

        # Add the assistant reply
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Display the new assistant message
        with st.chat_message("assistant"):
            st.write(assistant_reply)

        # Force re-run to show updated chat
        try:
            st.experimental_rerun()
        except:
            pass
