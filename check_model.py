
import pandas as pd
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA


# Load the trained model
model = xgb.XGBRegressor()
model.load_model("models/xgboost_solubility_model.json")

# Create ethanol from its SMILES
smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)

# Calculate the seven model features
descriptors = {
    "MolecularWeight": Descriptors.MolWt(mol),
    "LogP": Crippen.MolLogP(mol),
    "TPSA": CalcTPSA(mol),
    "HBD": Lipinski.NumHDonors(mol),
    "HBA": Lipinski.NumHAcceptors(mol),
    "RotatableBonds": Lipinski.NumRotatableBonds(mol),
    "RingCount": Descriptors.RingCount(mol),
}

X = pd.DataFrame([descriptors])

print("\n--- ETHANOL FEATURES ---")
print(X.to_string(index=False))

print("\n--- MODEL FEATURE NAMES ---")
print(model.get_booster().feature_names)

print("\n--- MODEL PREDICTION ---")
print(model.predict(X)[0])

print("\n--- ETHANOL IN TRAINING DATA ---")
df = pd.read_csv("data/processed/ESOL_clean.csv")

ethanol = df[df["canonical_smiles"] == "CCO"]
print(ethanol[["canonical_smiles", "logS"]].to_string(index=False))

if ethanol.empty:
    print("Ethanol was not found in the cleaned dataset.")
