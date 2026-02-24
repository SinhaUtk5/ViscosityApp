import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="UH-Viscosity Calculator", layout="wide")

# -----------------------------
# Title / Header (similar to Shiny)
# -----------------------------
st.title("UH-Viscosity Calculator")
st.subheader("A product of University of Houston")
st.markdown(
    'Department of Petroleum Engineering: "Interaction of Phase Behavior and Flow in Porous Media (IPBFPM) Consortium"'
)
st.markdown("**Based on the work shown in:**")
st.markdown(
    "- Sinha, U., Dindoruk, B., & Soliman, M. (2022). Physics Augmented Correlations and Machine Learning Methods "
    "To Accurately Calculate Dead Oil Viscosity Based on The Available Inputs. SPE Journal. (SPE-209610-PA)"
)

st.markdown(
    "Calculates the Viscosity (cp) of dead oil using Molecular Weight of Stock Tank Oil (MW), API and Temperature "
    "of Interest (°C) using XGB Method."
)

# Optional: links
st.markdown("1) Example input CSV template: (your link here)")
st.markdown("2) Short help video (no sound): (your link here)")

st.divider()

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "Viscosity_XGB_L1out.json"  # use your converted JSON


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
    """
    Mirrors your R transformations:

      T      <- IRIS[1:n,2]
      MW     <- IRIS[1:n,3]
      API    <- IRIS[1:n,4]

      SG     <- 141.5/(131.5+API)
      KW     <- 4.5579*(MW^0.15178)*(SG^-0.84573)
      LOGAPI <- log10(API)
      LOGT   <- log10(T)
      mult   <- log10(MW)/log10(API)

    Features used for prediction: (T, KW, MW, SG, API, LOGAPI, LOGT, mult)
    """
    if df.shape[1] < 4:
        raise ValueError(
            "Input CSV must have at least 4 columns (positional), like your Shiny app:\n"
            "col1=visc-like input (not used for prediction), col2=T, col3=MW, col4=API"
        )

    # Match R's positional indexing (R is 1-based; Python is 0-based)
    T = df.iloc[:, 1].astype(float)
    MW = df.iloc[:, 2].astype(float)
    API = df.iloc[:, 3].astype(float)

    SG = 141.5 / (131.5 + API)
    KW = 4.5579 * (MW**0.15178) * (SG ** (-0.84573))
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

    # Force exact order (critical for correctness)
    return X[FEATURE_COLS]


def predict_viscosity_cp(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    dtest = xgb.DMatrix(X.to_numpy())
    y = booster.predict(dtest)

    # invert transform exactly: 10^(10^(y)) - 1
    visc_cp = (10.0 ** (10.0**y)) - 1.0

    out = df.copy()
    out["Visc_predicted_cp"] = visc_cp
    return out


# -----------------------------
# UI: upload CSV, show table, download
# -----------------------------
uploaded = st.file_uploader("Upload the input CSV file here", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to see predictions.")
    st.stop()

try:
    input_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

try:
    result_df = predict_viscosity_cp(input_df)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.subheader("Results")
st.dataframe(result_df, use_container_width=True)

csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download the viscosity(cp) results",
    data=csv_bytes,
    file_name=f"Dead_Oil_Viscosity_Results-{pd.Timestamp.today().date()}.csv",
    mime="text/csv",
)
