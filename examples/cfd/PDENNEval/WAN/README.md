# WAN

## Poisson equation

### Part 1: high dimension problems

#### 1. training codes

```python
python WAN_Poisson-PH.py --d 2 --i 15000 --s 0 --b 1000
# --d the dimension of specific problem
# --i the epochs
# --s seed
# --b the weight of the boundary loss 
```

#### 2. testing codes

- L2RE: the resulting file 'metric-2D.csv' stores it.
- mERR: the resulting file 'metric-2D.csv' stores it.
- the infer time: the resulting file 'metric-2D.csv' stores it.
- the training time: File 'main.ipynb' computes training time based the input 'epoch_time-2D.csv' .
- the convergence time: File 'main.ipynb' computes convergence time based the input 'epoch_time-2D.csv' .

Other dimension problems can  refer the situation in 2D.

- **Part 2 and 3 is the same as Part 1 in testing codes.**

### Part 2: Singularity problem

#### 1. training codes

```python
python WAN_Poisson-PS.py --s 0
```

- Metrics accessed thought the similar method as in the Part1 of Function Learning-based NN Methods.

