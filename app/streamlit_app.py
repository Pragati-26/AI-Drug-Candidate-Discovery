
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Drug Candidate Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project structure:
# AI_Drug_Candidate_Discovery/
# ├── app/streamlit_app.py
# ├── models/
# └── results/candidates/

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_solubility_model.json"
SCALER_PATH = PROJECT_ROOT / "models" / "descriptor_scaler.pkl"
CANDIDATES_DIR = PROJECT_ROOT / "results" / "candidates"

RANKED_CSV = CANDIDATES_DIR / "top_10_ranked_candidates.csv"
FALLBACK_RANKED_CSV = CANDIDATES_DIR / "top_10_candidates.csv"
COMPARISON_CSV = CANDIDATES_DIR / "best_candidate_comparison.csv"
SAMPLING_CSV = CANDIDATES_DIR / "candidate_pool_sampling_chain.csv"

DESCRIPTOR_COLUMNS = [
    "MolecularWeight",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
]


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model and fitted scaler."""
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_csv(path_string):
    """Load a CSV file and cache its contents."""
    return pd.read_csv(path_string)


def find_existing_file(*paths):
    """Return the first existing file from a list of paths."""
    for path in paths:
        if path.exists():
            return path
    return None


def load_optional_csv(path, fallback=None):
    """Load an optional CSV without crashing the application."""
    selected_path = find_existing_file(path, fallback) if fallback else (
        path if path.exists() else None
    )

    if selected_path is None:
        return pd.DataFrame()

    try:
        return load_csv(str(selected_path))
    except Exception as exc:
        st.warning(f"Could not load {selected_path.name}: {exc}")
        return pd.DataFrame()


# ============================================================
# 3. MOLECULAR DESCRIPTORS
# ============================================================

def calculate_descriptors(smiles):
    """Convert a valid SMILES string into the seven model features."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            "Invalid SMILES. Please enter a chemically valid molecule."
        )

    descriptors = {
        "MolecularWeight": Descriptors.MolWt(molecule),
        "LogP": Crippen.MolLogP(molecule),
        "TPSA": rdMolDescriptors.CalcTPSA(molecule),
        "HBD": Lipinski.NumHDonors(molecule),
        "HBA": Lipinski.NumHAcceptors(molecule),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(
            molecule
        ),
        "RingCount":Lipinski.RingCount(molecule),
    }

    return molecule, pd.DataFrame(
        [[descriptors[column] for column in DESCRIPTOR_COLUMNS]],
        columns=DESCRIPTOR_COLUMNS,
    ), descriptors


def calculate_lipinski_violations(descriptors):
    """
    Count violations of the conventional Lipinski thresholds:
    MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.
    """
    violations = 0

    if descriptors["MolecularWeight"] > 500:
        violations += 1
    if descriptors["LogP"] > 5:
        violations += 1
    if descriptors["HBD"] > 5:
        violations += 1
    if descriptors["HBA"] > 10:
        violations += 1

    return violations


def get_column(df, possible_names):
    """Find a column using case-insensitive alternative names."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


# ============================================================
# 4. PAGE HEADER AND SIDEBAR
# ============================================================

st.title("🧬 AI-Driven Drug Candidate Discovery")

st.markdown(
    """
    Explore molecular solubility predictions, compare ranked drug
    candidates, and analyze candidate-pool sampling using a trained
    machine-learning model.
    """
)

st.sidebar.title("Navigation")
st.sidebar.caption("AI-powered molecular analysis")

st.sidebar.markdown(
    """
    **Project components**
    - Molecular descriptor calculation
    - XGBoost solubility prediction
    - Lipinski rule assessment
    - Ranked candidate exploration
    - Sampling-chain analysis
    - Candidate comparison
    """
)

st.sidebar.divider()
st.sidebar.caption("Research and educational use only.")


# ============================================================
# 5. VERIFY REQUIRED FILES
# ============================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append(str(MODEL_PATH.relative_to(PROJECT_ROOT)))

if not SCALER_PATH.exists():
    missing_files.append(str(SCALER_PATH.relative_to(PROJECT_ROOT)))

if missing_files:
    st.error(
        "Required model files are missing. Check your project structure."
    )

    for filename in missing_files:
        st.code(filename)

    st.info(
        "Place the trained model JSON and fitted scaler pickle in "
        "the models/ directory."
    )

    st.stop()

try:
    model, scaler = load_model()
except Exception as exc:
    st.error(f"Could not load the model or scaler: {exc}")
    st.stop()


# ============================================================
# 6. LOAD RESULT FILES
# ============================================================

ranked_df = load_optional_csv(RANKED_CSV, FALLBACK_RANKED_CSV)
comparison_df = load_optional_csv(COMPARISON_CSV)
sampling_df = load_optional_csv(SAMPLING_CSV)

ranked_smiles_col = get_column(
    ranked_df, ["SMILES", "smiles", "Canonical_SMILES"]
) if not ranked_df.empty else None

sampling_smiles_col = get_column(
    sampling_df, ["SMILES", "smiles", "Canonical_SMILES"]
) if not sampling_df.empty else None


# ============================================================
# 7. OVERVIEW METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Prediction Model", "XGBoost")
metric2.metric(
    "Ranked Candidates",
    len(ranked_df) if not ranked_df.empty else 0,
)
metric3.metric(
    "Sampling Steps",
    len(sampling_df) if not sampling_df.empty else 0,
)
metric4.metric(
    "Model Features",
    len(DESCRIPTOR_COLUMNS),
)

st.divider()


# ============================================================
# 8. APPLICATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔬 Molecule Prediction",
        "🏆 Ranked Candidates",
        "📊 Sampling Analysis",
        "⚖️ Candidate Comparison",
    ]
)


# ============================================================
# TAB 1: MOLECULE PREDICTION
# ============================================================

with tab1:
    st.header("Molecular Solubility Prediction")

    st.write(
        "Enter a molecule as a SMILES string to calculate its "
        "descriptors, estimate its solubility, and assess Lipinski rules."
    )

    with st.expander("What is SMILES?", expanded=False):
        st.write(
            "SMILES is a text representation of a molecular structure. "
            "For example, ethanol can be represented as CCO."
        )

    with st.form("prediction_form"):
        smiles_input = st.text_input(
            "Enter molecular SMILES",
            value="CCO",
            help="Example: CCO represents ethanol.",
        )

        submitted = st.form_submit_button(
            "Predict Molecular Properties",
            type="primary",
        )

    if submitted:
        try:
            molecule, descriptor_df, descriptor_values = (
                calculate_descriptors(smiles_input.strip())
            )

            scaled_features = scaler.transform(descriptor_df)

            prediction = float(
                model.predict(scaled_features)[0]
            )

            violations = calculate_lipinski_violations(
                descriptor_values
            )

            st.subheader("Prediction Results")

            result1, result2, result3 = st.columns(3)

            result1.metric(
                "Predicted logS",
                f"{prediction:.4f}",
            )

            result2.metric(
                "Molecular Weight",
                f"{descriptor_values['MolecularWeight']:.2f} g/mol",
            )

            result3.metric(
                "Lipinski Violations",
                violations,
            )

            st.caption(
                "logS is the model's predicted logarithmic solubility "
                "value. Its interpretation depends on the target's "
                "definition and units in the training dataset."
            )

            st.subheader("Calculated Molecular Descriptors")

            display_descriptors = descriptor_df.T.reset_index()
            display_descriptors.columns = [
                "Descriptor",
                "Value",
            ]

            st.dataframe(
                display_descriptors,
                width="stretch",
                hide_index=True,
            )

            st.subheader("Lipinski Rule-of-Five Assessment")

            rule_checks = [
                (
                    "Molecular weight ≤ 500",
                    descriptor_values["MolecularWeight"] <= 500,
                ),
                (
                    "LogP ≤ 5",
                    descriptor_values["LogP"] <= 5,
                ),
                (
                    "Hydrogen-bond donors ≤ 5",
                    descriptor_values["HBD"] <= 5,
                ),
                (
                    "Hydrogen-bond acceptors ≤ 10",
                    descriptor_values["HBA"] <= 10,
                ),
            ]

            for label, passed in rule_checks:
                if passed:
                    st.success(f"✓ {label}")
                else:
                    st.warning(f"✗ {label}")

            st.info(
                f"SMILES: {smiles_input.strip()}\n\n"
                f"Canonical SMILES: "
                f"{Chem.MolToSmiles(molecule)}"
            )

            prediction_record = pd.DataFrame(
                [
                    {
                        "SMILES": smiles_input.strip(),
                        "Canonical_SMILES": Chem.MolToSmiles(molecule),
                        **descriptor_values,
                        "Predicted_logS": prediction,
                        "Lipinski_Violations": violations,
                    }
                ]
            )

            st.download_button(
                "Download Prediction as CSV",
                data=prediction_record.to_csv(index=False).encode("utf-8"),
                file_name="molecule_prediction.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ============================================================
# TAB 2: RANKED CANDIDATES
# ============================================================

with tab2:
    st.header("Ranked Drug Candidates")

    if ranked_df.empty:
        st.warning(
            "No ranked-candidate CSV was found. Add "
            "top_10_ranked_candidates.csv or top_10_candidates.csv "
            "to results/candidates/."
        )
    else:
        st.write(
            "Review the saved candidate ranking generated by your "
            "project pipeline."
        )

        search_text = st.text_input(
            "Search candidate SMILES",
            key="ranked_search",
        )

        filtered_df = ranked_df.copy()

        if search_text and ranked_smiles_col:
            filtered_df = filtered_df[
                filtered_df[ranked_smiles_col]
                .astype(str)
                .str.contains(search_text, case=False, na=False)
            ]

        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Ranked Candidates",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        score_col = get_column(
            filtered_df,
            [
                "Solubility_Score",
                "Final_Score",
                "Score",
                "Predicted_logS",
            ],
        )

        if score_col and not filtered_df.empty:
            chart_df = filtered_df.copy()
            chart_df[score_col] = pd.to_numeric(
                chart_df[score_col], errors="coerce"
            )
            chart_df = chart_df.dropna(subset=[score_col])

            if not chart_df.empty:
                st.subheader(f"Candidate {score_col} Comparison")

                if ranked_smiles_col:
                    chart_df = chart_df.set_index(ranked_smiles_col)

                st.bar_chart(chart_df[score_col])


# ============================================================
# TAB 3: SAMPLING ANALYSIS
# ============================================================

with tab3:
    st.header("Candidate-Pool Sampling Analysis")

    if sampling_df.empty:
        st.warning(
            "Sampling results were not found. Add "
            "candidate_pool_sampling_chain.csv to results/candidates/."
        )
    else:
        st.write(
            "Inspect the saved sampling chain, including score changes "
            "and accepted proposals when those columns are available."
        )

        step_col = get_column(
            sampling_df, ["Step", "Iteration", "Iteration_Number"]
        )

        score_col = get_column(
            sampling_df,
            [
                "Solubility_Score",
                "Final_Score",
                "Score",
                "Predicted_logS",
            ],
        )

        accepted_col = get_column(
            sampling_df, ["Accepted", "Is_Accepted", "Acceptance"]
        )

        sampling_metric1, sampling_metric2, sampling_metric3 = (
            st.columns(3)
        )

        sampling_metric1.metric(
            "Recorded Steps",
            len(sampling_df),
        )

        if sampling_smiles_col:
            sampling_metric2.metric(
                "Unique Molecules",
                sampling_df[sampling_smiles_col].nunique(),
            )
        else:
            sampling_metric2.metric(
                "Unique Molecules",
                "N/A",
            )

        if accepted_col:
            accepted_values = (
                sampling_df[accepted_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            sampling_metric3.metric(
                "Accepted Proposals",
                int(accepted_values.sum()),
            )
        else:
            sampling_metric3.metric(
                "Accepted Proposals",
                "N/A",
            )

        if score_col:
            plot_df = sampling_df.copy()

            plot_df[score_col] = pd.to_numeric(
                plot_df[score_col], errors="coerce"
            )

            if step_col:
                plot_df[step_col] = pd.to_numeric(
                    plot_df[step_col], errors="coerce"
                )
                plot_df = plot_df.dropna(
                    subset=[step_col, score_col]
                )
                plot_df = plot_df.sort_values(step_col)
                plot_df = plot_df.set_index(step_col)
            else:
                plot_df = plot_df.dropna(subset=[score_col])
                plot_df = plot_df.reset_index(drop=True)
                plot_df.index = np.arange(1, len(plot_df) + 1)
                plot_df.index.name = "Step"

            if not plot_df.empty:
                st.subheader("Score Across Sampling Steps")
                st.line_chart(plot_df[score_col])

        st.subheader("Sampling Chain Records")

        st.dataframe(
            sampling_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Sampling Chain",
            data=sampling_df.to_csv(index=False).encode("utf-8"),
            file_name="candidate_pool_sampling_chain.csv",
            mime="text/csv",
        )

        st.caption(
            "The interface visualizes saved results; it does not rerun "
            "the sampling algorithm. Describe the method according to "
            "the actual implementation used to generate this CSV."
        )


# ============================================================
# TAB 4: CANDIDATE COMPARISON
# ============================================================

with tab4:
    st.header("Candidate Comparison")

    if comparison_df.empty:
        st.warning(
            "Candidate comparison data was not found. Add "
            "best_candidate_comparison.csv to results/candidates/."
        )
    else:
        st.write(
            "Compare the candidate records and properties saved by "
            "your analysis pipeline."
        )

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Candidate Comparison",
            data=comparison_df.to_csv(index=False).encode("utf-8"),
            file_name="best_candidate_comparison.csv",
            mime="text/csv",
        )

        comparison_smiles_col = get_column(
            comparison_df,
            ["SMILES", "smiles", "Canonical_SMILES"],
        )

        if comparison_smiles_col:
            selected_smiles = st.selectbox(
                "Inspect a candidate",
                comparison_df[comparison_smiles_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist(),
            )

            st.code(selected_smiles, language="text")

            try:
                molecule = Chem.MolFromSmiles(selected_smiles)

                if molecule is not None:
                    from rdkit.Chem import Draw

                    image = Draw.MolToImage(
                        molecule,
                        size=(400, 300),
                    )

                    st.image(
                        image,
                        caption="Selected molecular structure",
                    )
            except Exception as exc:
                st.caption(
                    f"Molecular structure visualization unavailable: {exc}"
                )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Drug Candidate Discovery | "
    "XGBoost molecular property prediction | "
    "Candidate ranking and sampling analysis"
)

st.warning(
    "Research prototype only. Predicted solubility and rule-based "
    "filters do not establish biological activity, drug safety, "
    "clinical effectiveness, or suitability for human use."
)
