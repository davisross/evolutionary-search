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
```
python3 search.py tsplib_files/berlin52.tsplib
```
