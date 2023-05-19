# contract and function Detection
## Prerequisites:

- Python3
- Tensorflow==1.15.0
- Numpy

## File:
For the data preprocessing:
- get contract & function relationship and feature: ./correlation
- ./correlation/sol: contract code
- ./correlation/sol_func_clean: Dangerous functions after processing
- ./correlation/similaritycalculate: Calculate the similarity between contracts, functions
- ./correlation/similarity: the similarity between contracts
- ./correlation/func_newvector:  get the feature of function
## DateSet
For the classification tasks:
- contract: ./data/graph_feature.txt
- contract_edge: ./data/graph_edge.txt
- function: ./data/func_feature.txt
- contract & function relationship: ./data/graph_index.txt ./data/func_index.txt

## Training:

contract detection: MTL_contract.py
contract & function detection: MTL_contractandfunction.py

*Ablation experiment*：MTL_contract_noenhancement.py MTL_contract_norelationship