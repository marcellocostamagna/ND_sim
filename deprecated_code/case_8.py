# Python script to generate the USRE fingerprint of molecules and compute their similarity

from rdkit import Chem
import similarity_3d as sim3d

molecules = Chem.SDMolSupplier('guac_sample_20.sdf')

# # benzene-benzene
# mol1 = molecules[0]
# mol2 = molecules[0]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarities, unpaired_atoms = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2)
# coefficients, penalty = sim3d.compute_coefficients(similarities, unpaired_atoms, mol1, mol2, fingerprint1, fingerprint2)
# similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
# print('Similarity between two identical molecules: {}'.format(similarity))


# benzene-triazine
mol1 = molecules[6]
mol2 = molecules[7]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarities = sim3d.compute_USRE_similarity_components(fingerprint1, fingerprint2, mol1, mol2)
coefficients, penalty = sim3d.compute_coefficients(similarities, mol1, mol2, fingerprint1, fingerprint2)
similarity = sim3d.compute_USRE_similarity(similarities, coefficients, penalty)
print('Similarity between two different molecules with the same elements: {}'.format(similarity))


