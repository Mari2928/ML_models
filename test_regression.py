import regression
from pathlib import Path

result = regression.mse_manual
weights = regression.manual_weights

output = Path("/home/mari.ashiga/ml_results.txt")
with open(output, 'a') as f:
    f.write(f"Weights: {weights}; MSE: {result}\n")
full_output = output.read_text()

print(full_output)