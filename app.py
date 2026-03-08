import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import json

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent Data Cleaning Assistant", layout="wide")

# --- Session State Management ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = []

def save_history():
    """Saves the current state of the dataframe to history."""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())

def undo_last_action():
    """Restores the last saved state."""
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        st.success("Undid last action.")
    else:
        st.warning("No history to undo.")

# --- AI Helper Functions ---
def get_gemini_response(prompt, api_key):
    """Interacts with the Gemini API with automatic model fallback."""
    try:
        genai.configure(api_key=api_key)
        
        # Try best free model first
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "404" in str(e):
                # Fallback to 1.5 Pro
                model = genai.GenerativeModel('gemini-1.5-pro-latest')
                response = model.generate_content(prompt)
                return response.text
            else:
                raise e
    except Exception as e:
        return e  # Return the actual Exception object if all models fail


# --- Sidebar UI ---
st.sidebar.title("Configuration")

# Check if API Key is in secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API Key loaded from Secrets ✅")
else:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])


if uploaded_file and st.session_state.df is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.toast("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# --- Main App Logic ---
st.title("🤖 Intelligent Data Cleaning Assistant")

if st.session_state.df is not None:
    # Top Bar: Undo & Reset
    col1, col2, col3 = st.columns([1, 1, 6])
    if col1.button("↩ Undo"):
        undo_last_action()
    if col2.button("🔄 Reset"):
        st.session_state.df = None
        st.session_state.history = []
        st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧹 Smart Cleaning", "🤖 AI Assistant", "💾 Export"])

    # --- Tab 1: Overview ---
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        st.subheader("Dataset Info")
        col_info1, col_info2 = st.columns(2)
        col_info1.write(f"**Rows:** {st.session_state.df.shape[0]}")
        col_info1.write(f"**Columns:** {st.session_state.df.shape[1]}")
        
        st.subheader("Missing Values")
        missing_data = st.session_state.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            st.bar_chart(missing_data)
        else:
            st.success("No missing values found!")

    # --- Tab 2: Smart Cleaning (Semantic Grouping) ---
    with tab2:
        st.subheader("Semantic Standardization")
        st.write("Group similar text values (e.g., 'USA', 'U.S.A.', 'United States') using AI.")
        
        # Select categorical columns
        cat_cols = st.session_state.df.select_dtypes(include=['object']).columns
        selected_col = st.selectbox("Select Column to Clean", cat_cols)
        
        if st.button("Analyze Unique Values"):
            if not api_key:
                st.error("Please enter an API Key in the sidebar.")
            else:
                unique_vals = st.session_state.df[selected_col].unique().tolist()
                st.write(f"Found {len(unique_vals)} unique values: {unique_vals[:10]}...")
                
                # Construct Prompt
                prompt = f"""
                Analyze the following list of unique values from a column named '{selected_col}'.
                Identify values that likely refer to the same entity (e.g., typos, abbreviations, case variations).
                Return a JSON mapping where the key is the *original value* and the value is the *standardized value*.
                Only include entries that need changing. Do NOT include unchanged values.
                
                List: {unique_vals}
                
                Return format: JSON only. No markdown. No explanations.
                Example output: {{"N. York": "New York", "NYC": "New York"}}
                """
                
                with st.spinner("AI is analyzing semantic similarities..."):
                    response = get_gemini_response(prompt, api_key)
                    
                    if isinstance(response, Exception):
                        st.error(f"API Error: {response}")
                        st.stop()
                    
                    try:
                        # Improved JSON extraction
                        import re
                        match = re.search(r"\{.*\}", response, re.DOTALL)

                        if match:
                            json_str = match.group(0)
                            mapping = json.loads(json_str)
                        else:
                            # Fallback if no braces found (unlikely but possible)
                            mapping = json.loads(response)
                        
                        if mapping:
                            st.write("Proposed Changes:")
                            st.json(mapping)
                            
                            if st.button("Apply Changes"):
                                save_history()
                                st.session_state.df[selected_col] = st.session_state.df[selected_col].replace(mapping)
                                st.success(f"Updated {len(mapping)} values!")
                                st.rerun()
                        else:
                            st.info("AI found no necessary changes.")
                    except json.JSONDecodeError:
                        st.error("Failed to parse AI response. The AI might have returned invalid JSON.")
                        st.expander("Raw Response").write(response)


    # --- Tab 3: AI Assistant (Natural Language to Code) ---
    with tab3:
        st.subheader("Natural Language Data Editing")
        st.write("Describe what you want to do (e.g., 'Remove rows where Age is missing' or 'Convert Date column to datetime format').")
        
        user_instruction = st.text_area("Your Instruction:")
        
        if st.button("Generate & Execute Code"):
            if not api_key:
                st.error("Please enter an API Key.")
            else:
                # Construct Prompt
                df_head = st.session_state.df.head().to_string()
                df_info = st.session_state.df.dtypes.to_string()
                
                prompt = f"""
                You are a Python Data Science Assistant. 
                I have a pandas DataFrame named `df`. 
                Here is the first 5 rows:
                {df_head}
                
                Here are the column types:
                {df_info}
                
                User Instruction: "{user_instruction}"
                
                Write Python code to execute this instruction on `df`.
                IMPORTANT:
                1. The code must be valid Python.
                2. It must modify `df` directly or assign the result back to `df`.
                3. Do NOT wrap the code in markdown blocks (no ```python).
                4. Do NOT include print statements or comments. Just the code.
                """
                
                with st.spinner("Generating code..."):
                    generated_code = get_gemini_response(prompt, api_key)
                    
                    if isinstance(generated_code, Exception):
                        st.error(f"API Error: {generated_code}")
                        st.stop()
                        
                    # Clean potential markdown
                    generated_code = generated_code.replace("```python", "").replace("```", "").strip()

                    
                    st.code(generated_code, language='python')
                    
                    try:
                        save_history()
                        # Create a local scope with 'df' and 'pd'
                        local_scope = {'df': st.session_state.df, 'pd': pd}
                        exec(generated_code, {}, local_scope)
                        st.session_state.df = local_scope['df']
                        st.success("Executed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing code: {e}")
                        undo_last_action() # Revert if error

    # --- Tab 4: Export ---
    with tab4:
        st.subheader("Download Cleaned Data")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
        )

else:
    st.info("Awaiting file upload. Please use the sidebar to start.")
