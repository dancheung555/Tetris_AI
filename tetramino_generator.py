import numpy as np
import pandas as pd
import random

random.seed(42)

def generate_n_bags(n_bags):
    tetramino_shapes = ["I", "J", "L", "O", "S", "T", "Z"]
    sequence = []
    for _ in range(n_bags):
        tetramino_shapes = random.sample(tetramino_shapes, k=7)
        # print(tetramino_shapes)
        sequence.append(tetramino_shapes)
    return sequence

def generate_next_bag():
    tetramino_shapes = ["I", "J", "L", "O", "S", "T", "Z"]
    return random.sample(tetramino_shapes, k=7)
    
def main():
    generate_next_bag()


if __name__ == "__main__":
    main()