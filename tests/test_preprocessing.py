# Import necessary packages and modules
import numpy as np
import tempfile
from rdkit import Chem
from ..source import pre_processing
import pytest
import os

# Create fixture for benzene molecule

@pytest.fixture
def benzene_molecule():
    # the following sdf information was extracted for benzene molecule
    benzene_sdf = """
 OpenBabel04252322153D

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.3829   -0.2211    0.0055 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5068   -1.3064   -0.0076 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8712   -1.0904   -0.0147 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3730    0.2110   -0.0046 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4968    1.2961    0.0109 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8812    1.0801    0.0137 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4567   -0.3898    0.0094 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8977   -2.3203   -0.0126 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5537   -1.9359   -0.0279 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4466    0.3794   -0.0086 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8877    2.3100    0.0204 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.5639    1.9256    0.0225 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  6  2  0  0  0  0
  1  7  1  0  0  0  0
  2  3  2  0  0  0  0
  2  8  1  0  0  0  0
  3  4  1  0  0  0  0
  3  9  1  0  0  0  0
  4  5  2  0  0  0  0
  4 10  1  0  0  0  0
  5  6  1  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
$$$$
"""
    return Chem.MolFromMolBlock(benzene_sdf, removeHs=False)

# Testing function collect_molecules_from_sdf
def test_collect_molecules_from_sdf(benzene_molecule):
    # Write benzene molecule to a temporary sdf file
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp:
        temp_name = temp.name
        writer = Chem.SDWriter(temp_name)
        writer.write(benzene_molecule)
        writer.close()
 
    molecules = pre_processing.collect_molecules_from_sdf(temp_name)

    # adjust these tests based on the contents of your test sdf file
    assert len(molecules) > 0
    assert isinstance(molecules[0], Chem.Mol)

# Testing function colslect_molecule_info
def test_collect_molecule_info(benzene_molecule):
    # Given
    # use benzene molecule as input

    # When
    info = pre_processing.collect_molecule_info(benzene_molecule)

    # Then
    assert isinstance(info, dict)
    assert set(info.keys()) == set(['coordinates', 'protons', 'delta_neutrons', 'formal_charges'])

# Testing function normalize_features
def test_normalize_features(benzene_molecule):
    # Given
    info = pre_processing.collect_molecule_info(benzene_molecule)

    # When
    normalized_info = pre_processing.normalize_features(info)

    # Then
    assert np.max(normalized_info['protons']) <= np.max(normalized_info['coordinates'])
    assert np.min(normalized_info['protons']) >= np.min(normalized_info['coordinates'])
    # Repeat this for 'neutrons' and 'formal_charges'

# Testing function taper_features
def test_taper_features(benzene_molecule):
    # Given
    info = pre_processing.collect_molecule_info(benzene_molecule)
    info = pre_processing.normalize_features(info)

    # When
    tapered_info = pre_processing.taper_features(info, np.tanh)  # for example, using np.tanh as tapering function

    # Then
    assert np.all(np.isfinite(tapered_info['protons']))  # no NaN or infinite values
    assert np.all(np.isfinite(tapered_info['delta_neutrons']))
    assert np.all(np.isfinite(tapered_info['formal_charges']))

# Testing function get_molecule_6D_datastructure
def test_get_molecule_6D_datastructure(benzene_molecule):
    # Given
    info = pre_processing.collect_molecule_info(benzene_molecule)
    info = pre_processing.normalize_features(info)
    info = pre_processing.taper_features(info, np.tanh)

    # When
    molecule_6D = pre_processing.get_molecule_6D_datastructure(info)

    # Then
    assert molecule_6D.shape[1] == 6  # Check if it's indeed a 6D data structure
    assert molecule_6D.shape[0] == benzene_molecule.GetNumAtoms()  # Check if the number of rows matches the number of atoms
