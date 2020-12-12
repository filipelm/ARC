#!/usr/bin/python

import os
import json
import re
import numpy as np
import skimage.measure as imgmeasure
from operator import le
from collections import defaultdict

"""
Student Name: Filipe Lima
ID Number: 20236042
Github: https://github.com/filipelm/ARC
"""

def solve_ded97339(x):
    """
    
    """
    y = np.zeros(x.shape, dtype=x.dtype)
    rmemory, cmemory = defaultdict(list), defaultdict(list)
    # Find the coordinates for all blocks in the grid.
    colored_blocks = list(zip(*np.where(x)))
    # Save the color of the block.
    color = x[colored_blocks[0]]
    # For each block coordinate, check whether another block
    # appears in the same row or in the same column. Draw a
    # straight line between the current block and any other
    # block that appears in the same row or column.
    for row, col in colored_blocks:
        # keep track of the row and column already seen.
        rmemory[row].append(col)
        cmemory[col].append(row)
        # Draw a line between the blocks in the edge of the row connection.
        y[row, min(rmemory[row]):max(rmemory[row]) + 1] = color
        # Draw a line between the blocks in the edge of the column connection.
        y[min(cmemory[col]):max(cmemory[col]) + 1, col] = color
    return y


def solve_b775ac94(x):
    """
    """
    def detect_structures(grid):
        labeled = imgmeasure.label(grid, connectivity=2)
        objects = [np.where(labeled == label) for label in np.unique(labeled) if label]
        for obj in objects:
            yield list(zip(*obj))

    def calculate_projection(structure):
        pass

    def replicate_structures(grid, structures):
        # Calculate coordinates for projections.
        projections = map(calculate_projection, structures)
        for color, projection in projections:
            # Apply projection to result.
            for coordinate in projection:
                grid[coordinate] = color
        return grid
    
    structures = detect_structures(x)
    major_structures = filter(lambda s: len(s) > 1, structures)
    y = replicate_structures(x, major_structures)
    return y


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__":
    main()
