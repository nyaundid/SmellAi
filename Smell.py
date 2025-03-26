# Import required libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
# Replace 'multi_labelled_smiles_odors.csv' with the path to your dataset
data = pd.read_csv("multi_labelled_smiles_odors.csv")

# Step 2: Filter for Stress-Relief Related Scents
# Example: Filtering for molecules with the "calming" descriptor
stress_relief_data = data[data['calming'] == 1]  # Modify column name as needed
print(f"Number of molecules with 'calming' odor descriptor: {len(stress_relief_data)}")

# Step 3: Feature Extraction from SMILES
# Define a function to extract molecular features using RDKit
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        return [None, None]
    return [Descriptors.MolWt(mol), Descriptors.NumRotatableBonds(mol)]

# Apply feature extraction to the dataset
stress_relief_data['features'] = stress_relief_data['SMILES'].apply(extract_features)
stress_relief_data = stress_relief_data.dropna(subset=['features'])  # Drop rows with invalid SMILES

# Separate features and labels
X = stress_relief_data['features'].tolist()
y = stress_relief_data['calming'].values  # Binary label for "calming"

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Step 7: Predict Stress-Relief Potential for a New Molecule
# Example: Predict the "calming" potential of a new SMILES string
new_smiles = "CC(C)C(=O)OC1=CC=CC=C1"  # Replace with a valid SMILES string
new_features = extract_features(new_smiles)

if None not in new_features:  # Ensure valid features
    prediction = model.predict([new_features])
    print(f"Predicted Stress-Relief Potential ('calming'): {'Yes' if prediction[0] == 1 else 'No'}")
else:
    print("Invalid SMILES string for prediction.")
