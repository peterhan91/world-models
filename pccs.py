import pandas as pd
import numpy as np
import warnings
from rdkit import Chem

def count_carbon_atoms(smiles, atom_name="C"):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mol = Chem.MolFromSmiles(smiles)
        if w:
            return np.nan
        if mol is None:  # Invalid SMILES string
            return np.nan
        
        atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == atom_name]
        return len(atoms)

df_pcc = pd.read_csv('../../Datasets/drugs/pccs-pubchem/test.tsv', sep='\t')
df_pcc_tr = pd.read_csv('../../Datasets/drugs/pccs-pubchem/train.tsv', sep='\t')

df_pcc_test = df_pcc[~df_pcc['Name'].isin(df_pcc_tr['Name'].tolist())]
df_pcc_tr['is_test'] = False
df_pcc_test['is_test'] = True

df_pcc_tr['No. C'] = df_pcc_tr['SMILES'].apply(count_carbon_atoms)
df_pcc_test['No. C'] = df_pcc_test['SMILES'].apply(count_carbon_atoms)
df_pcc_tr['No. N'] = df_pcc_tr['SMILES'].apply(lambda x: count_carbon_atoms(x, atom='N'))
df_pcc_test['No. N'] = df_pcc_test['SMILES'].apply(lambda x: count_carbon_atoms(x, atom='N'))
df_pcc_tr['No. O'] = df_pcc_tr['SMILES'].apply(lambda x: count_carbon_atoms(x, atom='O'))
df_pcc_test['No. O'] = df_pcc_test['SMILES'].apply(lambda x: count_carbon_atoms(x, atom='O'))

df_pcc_tr = df_pcc_tr.dropna()
df_pcc_test = df_pcc_test.dropna()
df_pcc_test = df_pcc_test.filter(['Name', 'SMILES', 'No. C', 'No. N', 'No. O', 'is_test'])
df_pcc_tr = df_pcc_tr.filter(['Name', 'SMILES', 'No. C', 'No. N', 'No. O', 'is_test'])
df = pd.concat([df_pcc_tr, df_pcc_test])

df.to_csv('./data/entity_datasets/pccs.csv', index=False)