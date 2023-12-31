# Python script to generate the USRE fingerprint of molecules and compute their similarity

from rdkit import Chem
import deprecated_code.similarity_3d as sim3d

molecules = Chem.SDMolSupplier('sample3d_optimized.sdf')
molecules_2 = Chem.SDMolSupplier('sample3d_optimized_switched.sdf')

# # Select two identical molecules from the list 
# mol1 = molecules[0]
# mol2 = molecules[0]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
# print('Similarity between two identical molecules: {}'.format(similarity))

# # Select two different molecules from the list with the same elements
# mol1 = molecules[0]
# mol2 = molecules[4]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
# print('Similarity between two different molecules with the same elements: {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 != 0, diff2 = 0)
# mol1 = molecules[3]
# mol2 = molecules[0]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
# print('Similarity between two different molecules with different elements (1): {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 = 0, diff2 != 0)
# mol1 = molecules[0]
# mol2 = molecules[6]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
# print('Similarity between two different molecules with different elements (2): {}'.format(similarity))


# # Select two different molecules from the list with different elements (diff1 & diff2 != 0)
# mol1 = molecules[3]
# mol2 = molecules[6]
# fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
# fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
# similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
# print('Similarity between two different molecules with different elements (3): {}'.format(similarity))


# Select identical molecules with oxygens and sulphurs switched
mol1 = molecules_2[0]
mol2 = molecules_2[1]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

# Select identical molecules with oxygens and sulphurs switched
mol1 = molecules_2[1]
mol2 = molecules_2[1]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

# Select identical molecules with oxygens and sulphurs switched (2)
mol1 = molecules_2[1]
mol2 = molecules_2[2]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))