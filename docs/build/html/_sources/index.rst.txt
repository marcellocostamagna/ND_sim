.. ND_sim documentation master file, created by
   sphinx-quickstart on Mon Nov 20 14:29:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ND_sim's documentation!
==================================

ND_sim: A Versatile Similarity Measure for Chemical Systems
-----------------------------------------------------------

ND_sim is a versatile similarity measure tool applicable to a wide range of chemical systems. 
It is developed to enable fast 3-D similarity measures across diverse chemicals and molecules. 
Based on the Ultrafast Shape Recognition (USR) method, ND_sim enhances robustness and versatility 
compared to similar previous methods.

In addition to considering the 3D coordinates of molecules, ND_sim incorporates additional 
features for each atom, making it a multidimensional (ND) similarity approach. While users 
have the flexibility to modify these features, the default configuration utilizes information 
based on the protons, neutrons, and formal charges of each atom.

Getting Started
---------------

Installing ND_sim
~~~~~~~~~~~~~~~~~

ND_sim can be easily installed using pip or conda. Choose the method that best suits your environment:

Using pip:

.. code-block:: bash 

    pip install nd_sim

Using conda:

.. code-block:: bash

    conda install nd_sim -c conda-forge


Understanding ND_sim
--------------------

Although the use of ND_sim can be pretty straightforward:

.. code-block:: python

      import nd_sim
      
      mol1 = load_molecules_from_sdf('mol1.sdf')
      mol2 = load_molecules_from_sdf('mol2.sdf')

      similarity = compute_similarity(mol1, mol2)

Some of the logic behind it can be complex at first.
To learn more about how ND_sim works and its underlying methodology, see the :doc:`overview of the method <detailed_overview>`.

Licensing
---------

ND_sim is licensed under the GNU Affero General Public License Version 3, 19 November 2007. For more details, 
see the LICENSE file in the `source code repository <https://github.com/marcellocostamagna/ND_sim>`_ or visit `GNU AGPL v3 License <https://www.gnu.org/licenses/agpl-3.0.html>`_.

Citing ND_sim
-------------

If you use ND_sim in your research, please cite it as follows:

[TODO: Add citation]

Contributing to ND_sim
----------------------

We welcome contributions to ND_sim! If you're interested in helping, 
please read our :doc:`Contributing Guidelines <CONTRIBUTING>` for information on how to get started.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules
   detailed_overview
   CONTRIBUTING

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
