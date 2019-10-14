# Multi-DTI
This project is about realizing Multi-DTI with python, see more information about Multi-DTI, you can look through《Drug Target Interaction Prediction using Multi-task Learning and Co-attention》 in 2019 BIBM.


## Environment
- Python 3.6.5
- numpy 1.14.3
- tensorflow 1.13.1
- keras 2.1.1

## Code and data
### code
- `model`: predict drug-target interactions (DTIs).
- `Input.py`: shuffle dataset and divide new data parts.
- `Metrics.py`: some useful metrics.
- `pre_process.py`: train an embedding for each element in drug sequence and protein sequence.
- `change_transformer.py`: build several keras models.


### `data/` directory
### `KIBA/` directory
- `drugs.txt`: list of drug chemical compound sequences.
- `proteins.txt`: list of protein amino acid sequences.
- `mat_drug_protein.txt`: Drug_Protein interaction matrix(downloaded from [here](https://pubs.acs.org/doi/suppl/10.1021/ci400709d))
**Note**: drugs, proteins, diseases and side-effects are organized in the same order across all files, including name lists, ID mappings and interaction/association matrices.


