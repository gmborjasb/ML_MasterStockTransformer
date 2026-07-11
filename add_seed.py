import json

seed_code = """import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)
"""

notebooks = [
    '02_Model_Architecture.ipynb',
    '03_Training_Loop.ipynb',
    '03B_Hyperparameter_Tuning.ipynb',
    '04_Evaluation_and_Visualizations.ipynb'
]

for nb_name in notebooks:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def set_seed' not in source:
                cell['source'] = [seed_code + "\n"] + cell['source']
            break
            
    with open(nb_name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
