import streamlit as st
import pandas as pd
import os
import subprocess
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit_ace import st_ace

st.set_page_config(page_title="AutoML Studio", page_icon="🚀", layout="wide")

st.title("🚀 AutoML Studio")
st.markdown("---")

# ---- STATE MANAGEMENT ----
if "trained" not in st.session_state:
    st.session_state["trained"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "preprocessor" not in st.session_state:
    st.session_state["preprocessor"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None
if "task" not in st.session_state:
    st.session_state["task"] = None
if "feature_names" not in st.session_state:
    st.session_state["feature_names"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

if "last_file" not in st.session_state:
    st.session_state["last_file"] = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        # Only load if it's a new file
        if st.session_state["last_file"] != uploaded_file.name:
            st.session_state["df"] = pd.read_csv(uploaded_file)
            st.session_state["last_file"] = uploaded_file.name
            st.session_state["trained"] = False # Reset training on new file
            st.success("File loaded successfully!")
        
        # Save current df to disk for backend CLI
        os.makedirs("data/raw", exist_ok=True)
        temp_path = "data/raw/streamlit_upload.csv"
        st.session_state["df"].to_csv(temp_path, index=False)
        
        if st.button("⚡ Run AutoML Pipeline"):
            with st.spinner("Analyzing and training..."):
                # Run CLI tool
                cmd = [
                    os.path.join(os.getcwd(), "venv", "bin", "python"),
                    "run_automl.py",
                    "--dataset", temp_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.session_state["trained"] = True
                    
                    # Load artifacts
                    with open("outputs/models/best_model.pkl", "rb") as f:
                        st.session_state["model"] = pickle.load(f)
                    with open("outputs/artifacts/preprocessor.pkl", "rb") as f:
                        st.session_state["preprocessor"] = pickle.load(f)
                    with open("outputs/artifacts/metrics.json", "r") as f:
                        report = json.load(f)
                        st.session_state["metrics"] = report["metrics"]
                        st.session_state["model_name"] = report["model_name"]
                        
                    # Extract task from stdout
                    for line in result.stdout.split("\n"):
                        if line.startswith("TASK_TYPE="):
                            st.session_state["task"] = line.split("=")[1]
                            
                    st.session_state["feature_names"] = st.session_state["df"].columns[:-1].tolist()
                    st.rerun()
                else:
                    st.error("Training failed!")
                    st.code(result.stderr)

with col2:
    if st.session_state["df"] is not None:
        st.subheader("📊 Dataset Preview")
        st.dataframe(st.session_state["df"].head(10))

        # ---- INTERACTIVE CODE SANDBOX ----
        st.markdown("---")
        st.subheader("💻 Interactive Code Sandbox")
        st.info("Directly manipulate the data `df` or plot using `plt`. Changes to `df` persist!")
        
        import sys
        import io
        import ast
        from contextlib import redirect_stdout

        code = st_ace(
            language="python",
            theme="monokai",
            height=250,
            value="# Example: df.drop(columns=['col'], inplace=True)\ndf.head()",
            key="ace_editor"
        )

        if st.button("🚀 Execute Code", key="run_sandbox"):
            output_container = st.container()
            try:
                # Prepare execution context
                ctx = {
                    "df": st.session_state["df"],
                    "pd": pd,
                    "np": np,
                    "plt": plt,
                    "sns": sns,
                    "st": st
                }
                
                # Capture stdout and evaluate last expression
                stdout_capture = io.StringIO()
                with redirect_stdout(stdout_capture):
                    # Parse the code into an AST
                    tree = ast.parse(code)
                    
                    # Split into statements and the last expression (if applicable)
                    last_expr = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = tree.body.pop()
                    
                    # Execute all but the last expression
                    if tree.body:
                        exec(compile(tree, filename="<ast>", mode="exec"), ctx)
                    
                    # If there was a last expression, evaluate it
                    evaluation_result = None
                    if last_expr:
                        evaluation_result = eval(compile(ast.Expression(last_expr.value), filename="<ast>", mode="eval"), ctx)

                # Show output
                with output_container:
                    st.markdown("**Output:**")
                    
                    # Show captured print statements
                    printed_text = stdout_capture.getvalue()
                    if printed_text:
                        st.code(printed_text)
                    
                    # Show last expression result (Jupyter style)
                    if evaluation_result is not None:
                        if isinstance(evaluation_result, (pd.DataFrame, pd.Series)):
                            st.dataframe(evaluation_result)
                        else:
                            st.write(evaluation_result)
                    
                    # Check for plots
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.close('all')
                
                # Update session state with modified df
                st.session_state["df"] = ctx["df"]
                st.success("Execution finished!")
                
            except Exception as e:
                st.error(f"Error executing code: {e}")
        
    if st.session_state["trained"]:
        st.markdown("---")
        st.subheader("🎯 Result: " + st.session_state.get("model_name", "Best Model"))
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Task Type", st.session_state["task"].capitalize())
        with m_col2:
            score = st.session_state["metrics"].get("accuracy") or st.session_state["metrics"].get("r2_score")
            st.metric("Primary Score", f"{score:.4f}")
            
        st.json(st.session_state["metrics"])

st.markdown("---")

if st.session_state["trained"]:
    st.subheader("🔮 Online Prediction")
    st.info("Enter values below to get real-time predictions from the trained model.")
    
    with st.form("prediction_form"):
        inputs = {}
        # Simple inputs for numerical/categorical
        # In a real GSoC project, we'd check types, but for now we use text_input/number_input
        input_cols = st.columns(3)
        for i, col in enumerate(st.session_state["feature_names"]):
            with input_cols[i % 3]:
                inputs[col] = st.text_input(col, value="0.0")
        
        submit = st.form_submit_state = st.form_submit_button("Predict")
        
        if submit:
            # Prepare input
            input_df = pd.DataFrame([inputs])
            # Convert numeric columns
            for col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    pass
            
            # Preprocess
            try:
                processed_input = st.session_state["preprocessor"].transform(input_df)
                prediction = st.session_state["model"].predict(processed_input)[0]
                
                st.success(f"### 🎯 Prediction: **{prediction}**")
                
                # SHAP Explanation
                st.markdown("---")
                st.subheader("🔍 Prediction Explanation (SHAP)")
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    
                    # Create an explainer
                    # For simplicity, we use KernelExplainer which works for any model but is slower
                    # For a professional project, we'd use TreeExplainer for RF/GBM
                    model_type = type(st.session_state["model"]).__name__
                    if "Forest" in model_type or "Boosting" in model_type:
                        explainer = shap.TreeExplainer(st.session_state["model"])
                    else:
                        explainer = shap.KernelExplainer(st.session_state["model"].predict, processed_input)

                    shap_values = explainer.shap_values(processed_input)
                    
                    # Plot
                    fig, ax = plt.subplots()
                    if isinstance(shap_values, list): # Classification
                        shap.initjs()
                        shap.force_plot(explainer.expected_value[0], shap_values[0], processed_input, matplotlib=True, show=False)
                    else:
                        shap.force_plot(explainer.expected_value, shap_values, processed_input, matplotlib=True, show=False)
                    
                    st.pyplot(plt.gcf())
                    st.write("The plot above shows how each feature contributed to the final prediction. Red bars increase the prediction, blue bars decrease it.")
                except Exception as shap_e:
                    st.warning(f"Could not generate SHAP explanation: {shap_e}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
