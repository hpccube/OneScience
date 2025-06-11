# DRM

## Poisson equation
### Part 1: high dimension problems

```python
python DRM_Poisson-Ph.py --dimension 3 --seed 0
# --dimension: dimension of the problem (default: 100)
# --seed: random seed (default: 0)
# --beta: weight of boundary loss (default: 1000)
```

### Part 2: Singularity problem

```python
python DRM_Poisson-Ps.py --dimension 2 --seed 0
```

### Results after training

All metrics are recorded in `{DIMENSION}DIM-DRM-{NUM_ITERATION}itr-{SAMPLE_FREQUENCY}N-.csv`.

- step: The current iteration number during the training process.
- L2error: The L2 norm error, measuring the difference between predicted and true values.
- MaxError: The maximum error observed between predicted and true values.
- loss: The total loss value, combining interior and boundary loss components.
- elapsed_time: The total time elapsed since the beginning of the training.
- epoch_time: The time taken to complete one training step.
- inference_time: The time taken to perform inference on the test data.
