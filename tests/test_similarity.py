import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from nd_sim import similarity

# Helper function to generate 3D conformer for a molecule
def generate_3d_coords(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

# Fixtures for RDKit Mol objects
@pytest.fixture
def ethanol_3d():
    mol = Chem.MolFromSmiles('CCO')
    return generate_3d_coords(mol)

@pytest.fixture
def charged_ethanol_3d():
    mol = Chem.MolFromSmiles('CCO')
    mol.GetAtomWithIdx(0).SetFormalCharge(1)
    return generate_3d_coords(mol)

@pytest.fixture
def ethane_3d():
    mol = Chem.MolFromSmiles('CC')
    return generate_3d_coords(mol)

def test_calculate_mean_absolute_difference():
    moments1 = [1, 2, 3, 4, 5]
    moments2 = [2, 3, 4, 5, 6]

    mean_absolute_difference = similarity.calculate_mean_absolute_difference(moments1, moments2)
    assert mean_absolute_difference == 1.0  # (1 + 1 + 1 + 1 + 1) / 5

def test_calculate_similarity_from_difference():
    partial_score = 0.5
    similarity_measure = similarity.calculate_similarity_from_difference(partial_score)
    assert similarity_measure == 2/3  # 1 / (1 + 0.5)

# Edge Cases
def test_calculate_mean_absolute_difference_empty_lists():
    moments1 = []
    moments2 = []
    
    with pytest.raises(ZeroDivisionError):
        similarity.calculate_mean_absolute_difference(moments1, moments2)

def test_calculate_mean_absolute_difference_different_lengths():
    moments1 = [1, 2, 3]
    moments2 = [4, 5]
    
    with pytest.raises(IndexError):
        similarity.calculate_mean_absolute_difference(moments1, moments2)

def test_compute_similarity_3d_mols(ethanol_3d, ethane_3d):
    # Similarity between identical molecules
    similarity_same = similarity.compute_similarity(ethanol_3d, ethanol_3d)
    assert similarity_same == 1 

    # Similarity between different molecules
    similarity_diff_1 = similarity.compute_similarity(ethanol_3d, ethane_3d)

    assert similarity_diff_1 < 1

def test_compute_similarity_charged_vs_neutral(ethanol_3d, charged_ethanol_3d, capsys):
    similarity_score = similarity.compute_similarity(ethanol_3d, charged_ethanol_3d, chirality=True)
    assert similarity_score < 1

    captured = capsys.readouterr()
    expected_warning = ("WARNING: Comparison between molecules of different dimensionality: "
                        "4 and 5.\n"
                        "The similarity score may not be accurate!")
    
    assert expected_warning in captured.out    
