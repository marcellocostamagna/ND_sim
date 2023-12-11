import unittest
import numpy as np
from rdkit import Chem
from nd_sim import utils

class TestUtilsFunctions(unittest.TestCase):
    
    def setUp(self):
        # create a sample molecule for testing
        self.molecule = Chem.MolFromSmiles("CCO")
        self.atom = self.molecule.GetAtomWithIdx(0)  # Carbon atom

    def test_extract_proton_number(self):
        self.assertEqual(utils.extract_proton_number(self.atom), np.sqrt(6))

    def test_extract_neutron_difference_from_common_isotope(self):
        # Carbon's most common isotope is 12C, which has 6 neutrons.
        # The difference between the neutrons of the most common isotope and itself should be 0
        self.assertEqual(utils.extract_neutron_difference_from_common_isotope(self.atom), 0)

    def test_extract_formal_charge(self):
        # Carbon atom in ethanol (CCO) has no formal charge.
        self.assertEqual(utils.extract_formal_charge(self.atom), 0)

    def test_compute_scaling_factor(self):
        molecule_data = np.array([[0, 0, 0], [2, 2, 2]])
        scaling_factor = utils.compute_scaling_factor(molecule_data)
        self.assertEqual(scaling_factor, 3.4641016151377544)  # sqrt(2^2 + 2^2 + 2^2)

    def test_compute_scaling_matrix(self):
        molecule_data = np.array([[0, 0, 0], [-2, 2, -3]])
        scaling_matrix = utils.compute_scaling_matrix(molecule_data)
        expected_matrix = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        self.assertTrue(np.array_equal(scaling_matrix, expected_matrix))

if __name__ == '__main__':
    unittest.main()
