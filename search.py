import math
import matplotlib.pyplot as plt
import random
import sys
import time


# Represent each city and store their index and coordinates
class City:
    def __init__(self, index, lat, lng):
        self.index = index
        self.lat = lat
        self.lng = lng


# Represent each path by storing their city list and fitness (total distance)
class Path:
    def __init__(self, path: list[City]):
        self.path = path
        self.fitness = self.calculate_fitness()

    # Calculate the total distance between every city in the list
    def calculate_fitness(self):
        distance = sum(euclidean_distance(self.path[i], self.path[i + 1]) for i in range(len(self.path) - 1))
        distance += euclidean_distance(self.path[-1], self.path[0])
        return distance


# Calculate the Euclidean distance between two cities
def euclidean_distance(city1: City, city2: City):
    return math.sqrt((city2.lat - city1.lat) ** 2 + (city2.lng - city1.lng) ** 2)


# Perform tournament selection on a given list of paths (individuals)
def perform_selection(paths: list[Path]):
    # Select 7 at random and return the top 2 fittest paths
    selection = random.sample(paths, 7)
    best_two = sorted(selection, key=lambda path: path.fitness)[:2]
    return best_two[0], best_two[1]


# Perform ordered crossover on two given paths
def ordered_crossover(parent_1: Path, parent_2: Path):
    size = len(parent_1.path)
    start, end = sorted(random.sample(range(size), 2))

    # Copy random section from parent 1 to the offspring
    child_path = [None] * size
    child_path[start:end] = parent_1.path[start:end]

    pos = end
    # Fill in remaining cities sourced from parent 2
    for city in parent_2.path:
        if city not in child_path:
            # If position exceeds array size, loop back to 0
            if pos >= size:
                pos = 0
            child_path[pos] = city
            pos += 1

    return Path(child_path)


# Perform modified ordered crossover on two given paths
def modified_ordered_crossover(parent_1: Path, parent_2: Path):
    size = len(parent_1.path)
    index = random.randint(1, size-1)

    # Construct the offspring path, fill with null values
    child_path = [None] * size
    left_p2 = parent_2.path[0:index]

    # Keep cities that appear in left side of parent_2
    for i in range(size):
        if parent_1.path[i] in left_p2:
            child_path[i] = parent_1.path[i]

    # Fill in blanks with right side of parent_2
    for i in range(size):
        if child_path[i] is None:
            child_path[i] = parent_2.path[index]
            index += 1

    return Path(child_path)    


# Perform crossover on two given paths, subject to a given rate
# If crossover is not performed, return parent 1
def perform_crossover(parent_1: Path, parent_2: Path, crossover_rate: float):
    if random.random() < crossover_rate:
        # 50/50 chance of which crossover algorithm is selected
        if random.random() < 0.5:
            return ordered_crossover(parent_1, parent_2)
        return modified_ordered_crossover(parent_1, parent_2)
    return parent_1


# Perform swap mutation on a given path
def swap_mutation(path: Path):
    i, j = random.sample(range(len(path.path)), 2)
    path.path[i], path.path[j] = path.path[j], path.path[i]
    path.fitness = path.calculate_fitness()


# Perform scramble mutation on a given path
def scramble_mutation(path: Path):
    i, j = sorted(random.sample(range(len(path.path)), 2))
    sublist = path.path[i:j]
    random.shuffle(sublist)
    path.path[i:j] = sublist
    path.fitness = path.calculate_fitness()


# Perform mutation on a given path, subject to a given rate
def perform_mutation(path: Path, mutation_rate: float):
    if random.random() < mutation_rate:
        # 50/50 chance of mutation algorithm used
        if random.random() < 0.5:
            swap_mutation(path)
        else:
            scramble_mutation(path)


# Run the genetic algorithm given several options and the list of cities
def run_genetic_algorithm(pop_size: int, max_iter: int, elitism_rate: int,
                          crossover_rate: float, mutation_rate: float, cities: list[City]):
    paths: list[Path] = []
    fitness_history: list[float] = []
    
    # Create initial random population
    for _ in range(pop_size):
        path = random.sample(cities, len(cities))
        paths.append(Path(path))

    # Perform iterations until limit is reached
    for iter in range(max_iter):
        new_paths: list[Path] = []
        
        # Elitism: let top n fittest progress
        new_paths.extend(sorted(paths, key=lambda path: path.fitness)[:elitism_rate])

        # Perform selection, crossover, and mutation (subject to respective rates)
        for _ in range(pop_size-elitism_rate):
            parent_1, parent_2 = perform_selection(paths)
            new_path = perform_crossover(parent_1, parent_2, crossover_rate)
            perform_mutation(new_path, mutation_rate)
            new_paths.append(new_path)

        # Log the shortest path (fittest individual) in the population
        fitness_history.append(min(new_paths, key=lambda path: path.fitness).fitness)
        # Overwrite old population with new population
        paths = new_paths

    # Find the shortest/fittest path and return with fitness history
    best_path = min(paths, key=lambda path: path.fitness)
    return best_path, fitness_history


# Parse a given TSPLIB file
def parse_tsp_file(filepath: str) -> list[City]:
    with open(filepath, 'r') as file:
        lines = file.readlines()
        file.close()

    cities = []
    parsing_coords = False

    for line in lines:
        # Remove whitespace from lines, they cause errors
        line = line.strip()

        # Exit when end of file is reached
        if line == "EOF":
            break

        if parsing_coords:
            values = line.split()
            index = int(values[0])
            lat = float(values[1])
            lng = float(values[2])
            cities.append(City(index, lat, lng))

        # Start reading coordinates when this line is reached
        if line == "NODE_COORD_SECTION":
            parsing_coords = True

    return cities


def main():
    # Parameters
    population_size = 200       # Size of the population
    maximum_iterations = 1000   # Maximum number of iterations performed
    elitism_rate = 10           # How many top paths should progress to the next round
    crossover_rate = 0.8        # Rate at which crossover is performed
    mutation_rate = 0.2         # Rate at which mutation is performed
    tsp_file = sys.argv[1]      # Path to TSPLIB file

    # Parse the input TSPLIB file and get a list of City objects
    cities = parse_tsp_file(tsp_file)

    # Run the genetic algorithm with the above parameters
    start_time = time.time()
    best_path, fitness_history = run_genetic_algorithm(population_size, maximum_iterations, elitism_rate,
                                                       crossover_rate, mutation_rate, cities)
    end_time = time.time()

    print(f'Genetic algorithm completed in {end_time-start_time:.3f} seconds.')
    print(f'The best path found has a fitness of {best_path.fitness:.6f}:')
    print([city.index for city in best_path.path])

    # Create fitness history graph
    plt.plot(fitness_history)
    plt.xlabel("Generations")
    plt.ylabel("Best fitness (distance)")
    plt.title("Fitness over generations")
    plt.show()


if __name__ == "__main__":
    main()
