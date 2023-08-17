# Python script to generate the USRE fingerprint of molecules and compute their similarity

from rdkit import Chem
import deprecated_code.similarity_3d as sim3d

molecules = Chem.SDMolSupplier('isomers_1.sdf')
#molecules = Chem.SDMolSupplier('isomers_mass.sdf')
#molecules = Chem.SDMolSupplier('isomers_try.sdf')

mol1 = molecules[0]
mol2 = molecules[0]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

mol1 = molecules[0]
mol2 = molecules[1]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

mol1 = molecules[0]
mol2 = molecules[2]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

mol1 = molecules[0]
mol2 = molecules[3]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

mol1 = molecules[0]
mol2 = molecules[4]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))

mol1 = molecules[0]
mol2 = molecules[5]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))


mol1 = molecules[0]
mol2 = molecules[6]
fingerprint1 = sim3d.generate_usre_fingerprint(mol1)
fingerprint2 = sim3d.generate_usre_fingerprint(mol2)
similarity = sim3d.compute_USRE_similarity(fingerprint1, fingerprint2, mol1, mol2)
print('Similarity between two identical molecules with oxygens and sulphurs switched: {}'.format(similarity))