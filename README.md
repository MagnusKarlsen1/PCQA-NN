# PCQA-NN

To create the dataset used in the paper, run the following Python code with the appropriate mesh size:

```python
mesh_size = [0.5, 1, 2]  # Dataset 1
mesh_size = np.arange(0.5, 2.1, 0.1)  # Dataset 2
```

Then generate the datasets using:

```bash
python Master_generator.py
```

This will create **Dataset 1 or 2** depending on the `mesh_size` range set inside the script.


**Note:** Requires **SolidWorks**.

If you only want to perform feature extraction on an `.xyz` file, use the function:

```python
GetVariables()
```

from the script:

```
geometric_functions.py
```

The XGBoost model and FLAX MLP model are available in the Jupyter Notebooks inside the folder `models`.

**Note:** The model isn't saved so a training of the models needs to be done using dataset 2.