# ND_sim: A Versatile Similarity Measure for Chemical Systems

ND_sim is a versatile tool for measuring similarity across a wide range of chemical systems. It enhances the robustness and versatility of the Ultrafast Shape Recognition (USR) method by incorporating multidimensional (ND) features for each atom, such as protons, neutrons, and formal charges.

## Getting Started

### Installing ND_sim

You can install ND_sim using either pip or conda:

```bash
pip install nd_sim
```
or 

```bash
conda install nd_sim -c conda-forge
```

### Basic Usage

Using ND_sim is straightforward:

```python
import nd_sim

mol1 = load_molecules_from_sdf('mol1.sdf')
mol2 = load_molecules_from_sdf('mol2.sdf')

similarity = compute_similarity(mol1, mol2)

```
For a detailed overview of ND_sim's methodology check our [documentation](https://marcellocostamagna.github.io/ND_sim/).


### Licensing

ND_sim is licensed under the [GNU Affero General Public License Version 3](https://www.gnu.org/licenses/agpl-3.0.html). For more details, see the LICENSE file.

### Citing ND_sim

If you use ND_sim in your research, please cite it as follows:

[TODO: Add citation]

### Contributing

Contributions to ND_sim are welcome! Please read our [Contributing Guidelines](https://marcellocostamagna.github.io/ND_sim/CONTRIBUTING.html) for information on how to get started.


