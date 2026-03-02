import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="UH-Viscosity Calculator", layout="wide")

# -----------------------------
# Title / Header
# -----------------------------
st.title("Dead Oil Viscosity Calculator")
st.markdown(
    'Product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium'
)
st.markdown("**Based on the work shown in:**")
st.markdown(
    "- Sinha, U., Dindoruk, B., & Soliman, M. (2022). Physics Augmented Correlations and Machine Learning Methods "
    "To Accurately Calculate Dead Oil Viscosity Based on The Available Inputs. SPE Journal. (SPE-209610-PA)"
)

st.markdown(
    "Calculates the Viscosity (cp) of dead oil using Molecular Weight of Stock Tank Oil (MW), API, and Temperature "
    "of Interest (°C) using XGB Method."
)

# Hyperlinks (as requested)
st.markdown(
    "1) **Download the example input CSV template**: "
    "[Click here](https://drive.google.com/file/d/1y-DxwgwosfZna6ip5-oezFzeiHjBpZtV/view?usp=drive_link)"
)
st.markdown(
    "2) **Watch short help video (no sound)**: "
    "[Click here](https://drive.google.com/file/d/1Gpd1ZlIimzP8EFscFj_Ys6WtGLtN3UBb/view?usp=drive_link)"
)

st.divider()

# -----------------------------
# Faculty & Contributor Section
# -----------------------------
APP_DIR = Path(__file__).resolve().parent

def show_resized_image(img_name: str, target_height: int):
    img_path = APP_DIR / img_name
    if not img_path.exists():
        st.warning(f"Missing image: {img_name}")
        return

    img = Image.open(img_path)

    # resize keeping aspect ratio, based on height
    w, h = img.size
    new_h = target_height
    new_w = int(w * (new_h / h))
    img_resized = img.resize((new_w, new_h))

    st.image(img_resized)

col1, col2 = st.columns(2)
TARGET_H = 200  # change to 160/180/220 as you like

with col2:
    show_resized_image("dindoruk_birol_2023_ns.png", TARGET_H)
    st.markdown(
        """
        **Dr. Birol Dindoruk**  
        Professor  
        Harold Vance Department of Petroleum Engineering,  
        Texas A&M University
        """
    )

with col1:
    show_resized_image("Utk.jpeg", TARGET_H)
    st.markdown(
        """
        **Utkarsh Sinha**  
        Volunteer Research Associate  
        Interaction of Phase-Behavior and Flow (IPB&F) Consortium
        """
    )

st.divider()
# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "Viscosity_XGB_L1out.json"

@st.cache_resource
def load_booster(model_path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

try:
    booster = load_booster(MODEL_PATH)
except Exception as e:
    st.error(
        f"Could not load model from: {MODEL_PATH}\n\n"
        f"Error: {e}\n\n"
        "Tip: Ensure the model file is present next to this app (or provide a correct path)."
    )
    st.stop()

# -----------------------------
# Feature engineering (match R code)
# -----------------------------
FEATURE_COLS = ["T", "KW", "MW", "SG", "API", "LOGAPI", "LOGT", "mult"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 4:
        raise ValueError(
            "Input CSV must have at least 4 columns (positional), like your Shiny app:\n"
            "col1=visc-like input (not used for prediction), col2=T, col3=MW, col4=API"
        )

    T = df.iloc[:, 1].astype(float)
    MW = df.iloc[:, 2].astype(float)
    API = df.iloc[:, 3].astype(float)

    SG = 141.5 / (131.5 + API)
    KW = 4.5579 * (MW ** 0.15178) * (SG ** (-0.84573))
    LOGAPI = np.log10(API)
    LOGT = np.log10(T)
    mult = np.log10(MW) / np.log10(API)

    X = pd.DataFrame(
        {
            "T": T,
            "KW": KW,
            "MW": MW,
            "SG": SG,
            "API": API,
            "LOGAPI": LOGAPI,
            "LOGT": LOGT,
            "mult": mult,
        }
    )
    return X[FEATURE_COLS]

def predict_viscosity_cp(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    dtest = xgb.DMatrix(X.to_numpy())
    y = booster.predict(dtest)
    visc_cp = (10.0 ** (10.0 ** y)) - 1.0

    out = df.copy()
    out["Visc_predicted_cp"] = visc_cp
    return out

# -----------------------------
# UI
# -----------------------------

# Keep data in session state so it persists after button clicks
if uploaded is not None:
    try:
        st.session_state["input_df"] = pd.read_csv(uploaded)
        st.success("CSV loaded. Click **Run prediction** to generate results.")
        st.dataframe(st.session_state["input_df"], use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to enable prediction.")

run_clicked = st.button("Run prediction", type="primary", disabled=("input_df" not in st.session_state))

if run_clicked:
    try:
        result_df = predict_viscosity_cp(st.session_state["input_df"])
        st.session_state["result_df"] = result_df
        st.success("Prediction complete.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

if "result_df" in st.session_state:
    st.subheader("Results")
    st.dataframe(st.session_state["result_df"], use_container_width=True)

    csv_bytes = st.session_state["result_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download the viscosity(cp) results",
        data=csv_bytes,
        file_name=f"Dead_Oil_Viscosity_Results-{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
    )









