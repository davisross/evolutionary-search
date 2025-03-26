# Evolutionary search (GAs)

This repo contains a Python script which implements a genetic algorithm and solves the travelling salesperson problem. It accepts a _.tsplib_ file as input. Examples of these can be found in the _[tsplib_files](tsplib_files)_ folder.

The script computes:
- Shortest path found and its fitness
- Time to compute
- Fitness plot over generations

And performs:
- Tournament selection
- Elitism
- Ordered crossover
- Modified ordered crossover
- Swap mutation
- Scramble mutation

### Prerequisites

```
pip3 install matplotlib
```

### Usage

To run the algorithm, specify the _TSPLIB_ file you would like to search:
```
python3 search.py tsplib_files/berlin52.tsplib
```

Parameters that control the search process are defined within the script's main function at [line 191](https://github.com/davisross/evolutionary-search/blob/main/search.py#L191).
