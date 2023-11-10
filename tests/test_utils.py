import unittest
import numpy as np
from rdkit import Chem
from similarity.source import utils

class TestUtilsFunctions(unittest.TestCase):
    
    def setUp(self):
        # create a sample molecule for testing
        self.molecule = Chem.MolFromSmiles("CCO")
        self.atom = self.molecule.GetAtomWithIdx(0)  # Carbon atom

    def test_extract_proton_number(self):
        self.assertEqual(utils.extract_proton_number(self.atom), 6)

    def test_extract_neutron_differencce(self):
        # Carbon usually has 6 neutrons for 12C isotope. Difference = 12 - 6 = 6
        self.assertEqual(utils.extract_neutron_difference(self.atom), 6)

    def test_extract_neutron_difference_from_common_isotope(self):
        # Carbon's most common isotope is 12C, which has 6 neutrons.
        # The difference between the neutrons of the most common isotope and itself should be 0
        self.assertEqual(utils.extract_neutron_difference_from_common_isotope(self.atom), 0)

    def test_extract_formal_charge(self):
        # Carbon atom in ethanol (CCO) has no formal charge.
        self.assertEqual(utils.extract_formal_charge(self.atom), 0)

    def test_taper_functions(self):
        self.assertEqual(utils.taper_p(10), np.sqrt(10))
        self.assertEqual(utils.taper_n(10), np.sign(10) * np.sqrt(10))
        self.assertEqual(utils.taper_c(0), 0)
        self.assertEqual(utils.taper_c(-5), -5)

    def test_normalize_feature_by_coordinate_range(self):
        feature_data = np.array([1, 2, 3])
        coordinates = np.array([[0, 0, 0], [2, 2, 2]])
        normalized_feature = utils.normalize_feature_by_coordinate_range(feature_data, coordinates)
        self.assertTrue(np.array_equal(normalized_feature, np.array([0., 1., 2.])))

    def test_normalize_feature_by_selected_axis(self):
        feature_data = np.array([1, 2, 3])
        coordinates = np.array([[0, 0, 0], [2, 4, 6]])
        normalized_feature = utils.normalize_feature_by_selected_axis(feature_data, coordinates, "smallest")
        self.assertTrue(np.array_equal(normalized_feature, np.array([0, 1, 2])), f"Expected [0.5, 1., 1.5], but got {normalized_feature}")

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
