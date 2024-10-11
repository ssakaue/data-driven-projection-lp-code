# Data-Driven Projection for Reducing Dimensionality of Linear Programs: Generalization Bound and Learning Methods

Official implementation of [Data-Driven Projection for Reducing Dimensionality of Linear Programs: Generalization Bound and Learning Methods]().

## Requirements
- gurobipy 10.0.1
- numpy 1.23.2
- pandas 1.4.4
- requests 2.28.1
- tqdm 4.64.1

## How to create LP instances

### Synthetic 
To create synthetic instances (packing, max-flow, and min-cost-flow), run the code in make_instance.ipynb. Created instances will be saved in "data" directory. 

### Netlib
1. Run get_netlib_data.py to fetch data from [Netlib](https://www.netlib.org/lp/data/), which will be saved in "netlib" directory.
2. Rung get_netlib_instance.py to obtain LP instances from fetched data, which will be saved in "data" directory. 

## How to train/test
For synthetic datasets: 
- train: `python3 train.py [data_name] [m] [n] [sigma]`
- test: `python3 test.py [data_name] [m] [n] [sigma]`

For netlib datasets: 
- train: `python3 train.py [data_name] [file_name]`
- test: `python3 test.py [data_name] [file_name]`

Learned projection matrices and results will be saved in "model" and "result" directories, respectively.

## How to plot results
Run the code in plot_result.ipynb.

## License
MIT