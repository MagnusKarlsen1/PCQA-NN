# PCQA-NN



To create the dataset used in the paper run the following codes, with the correct mesh size:

```python
mesh_size = [0.5, 1, 2] # Dataset 1
mesh_size = np.arange(0.5, 2.1, 0.1) # Dataset 2

Code to generate the datasets:

```bash
python Master_generator.py
