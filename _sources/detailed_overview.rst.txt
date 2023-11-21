Overview
========

ND_sim represents an approach to molecular similarity assessment, 
leveraging a multidimensional array to encapsulate both spatial and feature-based characteristics of molecules.
The method is grounded in a robust and deterministic process, ensuring precision and consistency in similarity measurements.

Initial Data Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Molecules are represented in an N-dimensional array, where the first three dimensions correspond to 3-D spatial coordinates (:func:`molecule_to_ndarray <nd_sim.pre_processing.molecule_to_ndarray>`).

- Additional features are integrated, enhancing the molecular description. In the default setting (:mod:`Utils <nd_sim.utils>`), these include:

    - Proton count, adjusted using a square root tapering function.
    - Neutron count difference from the most common isotope, also tapered by a square root function (with sign adjustment).
    - Formal charge, incorporated without tapering.

Principal Component Analysis (PCA) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The method applies PCA to the N-dimensional (6-D in default mode) molecular representation, extracting principal components of the molecule in the N-D space (:func:`compute_pca_using_covariance <nd_sim.pca_transform.compute_pca_using_covariance>`).

- Orientation of eigenvectors is determined rigorously. The sign of each eigenvector is set based on the maximum projection of the data onto that eigenvector. This ensures a deterministic and unambiguous assignment of orientation (:func:`adjust_eigenvector_signs <nd_sim.pca_transform.adjust_eigenvector_signs>`).

Fingerprint Construction
~~~~~~~~~~~~~~~~~~~~~~~~
- Post-PCA, the method constructs a molecular fingerprint (:mod:`Fingerprint <nd_sim.fingerprint>`). This involves selecting reference points corresponding to each principal component and the geometric center of the molecule.
- The distance of each reference point from the center can be adjusted. By default, it is set to the maximum coordinate value in that dimension.
- For each reference point, distances to all atoms are calculated, resulting in a set of distance distributions.
- From each distribution, three statistical moments are computed: mean, standard deviation, and skewness. These values are compiled into a list, forming the comprehensive fingerprint of the molecule.

Similarity Measurement
~~~~~~~~~~~~~~~~~~~~~~
- Molecular similarity is quantified using the inverse Manhattan distance between the fingerprints of two molecules (:mod:`Similarity <nd_sim.similarity>`). This metric provides a straightforward yet effective measure of similarity, capturing both spatial and feature-based nuances.


Examples
~~~~~~~~

The ND_sim method can be directly used to compute the similarity between two RDKit molecules:

.. code-block:: python

    import nd_sim

    mol1 = nd_sim.load_molecules_from_sdf('mol1.sdf')
    mol2 = nd_sim.load_molecules_from_sdf('mol2.sdf')

    similarity = nd_sim.compute_similarity(mol1, mol2)

In this example, :func:`compute_similarity()` is used with its default values:

.. code-block:: python

    compute_similarity(mol1, mol2, features=DEFAULT_FEATURES, scaling_method='matrix', removeHs=False, chirality=False)

But, if desired, the method can be "deconstructed" into its more elementary steps. Here, we first compute the fingerprint and then the similarity score:

.. code-block:: python

    import nd_sim

    mol1 = nd_sim.load_molecules_from_sdf('mol1.sdf')
    mol2 = nd_sim.load_molecules_from_sdf('mol2.sdf')

    fingerprint1 = nd_sim.generate_nd_molecule_fingerprint(mol1)
    fingerprint2 = nd_sim.generate_nd_molecule_fingerprint(mol2)

    similarity = nd_sim.compute_similarity_score(fingerprint1, fingerprint2)

In this case, the function :func:`generate_nd_molecule_fingerprint()` is used with its default values:

.. code-block:: python

    generate_nd_molecule_fingerprint(molecule, features=DEFAULT_FEATURES, scaling_method='matrix', scaling_value=None, chirality=False, removeHs=False)

An even more "exploded" example:

.. code-block:: python

    import nd_sim

    # Molecules from file
    mol1 = nd_sim.load_molecules_from_sdf('mol1.sdf')
    mol2 = nd_sim.load_molecules_from_sdf('mol2.sdf')
   
    # PCA
    mol1_transform = nd_sim.compute_pca_using_covariance(mol1)
    mol2_transform = nd_sim.compute_pca_using_covariance(mol2)
   
    # (Optional) Possibility to define personalized scaling for reference points' positions
    # to insert in the calculation of the fingerprint
 
    # Fingerprints
    fp1 = nd_sim.generate_molecule_fingerprint(mol1_transform) 
    fp2 = nd_sim.generate_molecule_fingerprint(mol2_transform)

    # Similarity
    similarity = nd_sim.compute_similarity_score(fp1, fp2)

This detailed step-by-step approach provides a deeper insight into the workings of the ND_sim method. By deconstructing the process, users can gain a better understanding of how each step contributes to the final similarity measurement. This can be particularly useful for debugging, optimizing, or simply gaining a more thorough understanding of the method's behavior with specific molecules. It allows for a granular inspection of the output at each stage, offering an opportunity to identify and analyze the characteristics of the molecules that are most influential in the similarity assessment.

Adding New Features
~~~~~~~~~~~~~~~~~~~

The ND_sim tool comes with its default features, but users have the flexibility to define new ones for their specific needs. New features must be capable of extracting or adding a property to each atom, and optionally, the value of this property can be scaled as desired. The function for obtaining the raw value of the property and the optional scaling function should be collected in a dictionary, as shown in the following example:

.. code-block:: python

    EXAMPLE = {
        'new_feature': [extract_new_feature, scale_new_feature],
    }

For comparison, here is the dictionary of the default features:

.. code-block:: python

    DEFAULT_FEATURES = {
        'protons': [extract_proton_number, taper_p],
        'delta_neutrons': [extract_neutron_difference_from_common_isotope, taper_n],
        'formal_charges': [extract_formal_charge, taper_c]
    }

For detailed insights into the implementation and management of these features within ND_sim, refer to the :mod:`Utils <nd_sim.utils>` module.

Chirality
~~~~~~~~~

ND_sim is capable of handling and distinguishing chiral molecules. However, this feature is not enabled by default, as it introduces additional complexity and potential reliability issues. For more detailed information on this aspect, please refer to our publication (TODO: add reference).

To consider chirality in your analysis, simply set the `chirality` flag to `True`. This can be done in either of the following ways:

When generating a fingerprint:

.. code-block:: python

    fingerprint = nd_sim.generate_nd_molecule_fingerprint(mol1, chirality=True)

Or when computing similarity:

.. code-block:: python

    compute_similarity(mol1, mol2, chirality=True)

Disclaimer
~~~~~~~~~~

Introducing chirality into the similarity measurement process can make the method less reliable, particularly when comparing molecules with differing dimensionality, such as a different number of principal components. An example of this might be comparing similar 3-D molecules where one has charges and the other is neutral. In such cases, the addition of chirality detection may further reduce the similarity score. For detailed explanations, please refer to our publication (TODO: add reference).

We recommend enabling chirality detection only in scenarios where molecules are unlikely to be described by different numbers of dimensions. However, it's important to note that this probability can never be completely eliminated, as some molecules might be planar, leading to dimensionality reduction after PCA. Therefore, if chirality is set to `True` and the dimensionality of the two molecules being compared differs, the method will issue a warning as follows:

.. code-block:: python

    "WARNING: Comparison between molecules of different dimensionality: {dimensionality1} and {dimensionality2}.\n"
                   "The similarity score may not be accurate!"


**IMPORTANT NOTE:**

   When the `chirality` parameter is set to `True`, both the :func:`compute_pca_using_covariance` and :func:`generate_nd_molecule_fingerprint` functions return an additional value – the dimensionality of the molecule. This change in return values is crucial to note, especially when these methods are used in a step-wise manner.

   The :func:`compute_similarity` function is designed to handle these additional return values correctly. It will process the dimensionality information and issue a warning if there is a mismatch in dimensionality between the two molecules being compared. This is particularly important because a difference in dimensionality can significantly impact the accuracy of the similarity score.

   If you are using :func:`compute_pca_using_covariance` or :func:`generate_nd_molecule_fingerprint` directly in your code, be prepared to handle an additional return value (the dimensionality) when `chirality` is `True`. This is especially relevant if you are integrating these functions into a larger workflow or using them in conjunction with other methods.

   For example, if you are performing PCA transformation step-by-step, you should modify your code to accommodate the additional dimensionality information. Similarly, when generating fingerprints, ensure that your code can handle the extra return value without errors.

   This change in the return structure is a direct consequence of enabling chirality detection, which adds a layer of complexity to the analysis but can provide more nuanced insights, especially for chiral molecules.
