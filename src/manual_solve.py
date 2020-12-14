#!/usr/bin/python

import os
import json
import re
import numpy as np
import skimage.measure
import scipy
from collections import defaultdict

"""
Student Name: Filipe Lima
ID Number: 20236042
Github: https://github.com/filipelm/ARC
"""


def solve_ded97339(x):
    """
    """
    y = np.copy(x)
    rmemory, cmemory = defaultdict(list), defaultdict(list)
    colored_blocks = list(zip(*np.where(x)))
    color = x[colored_blocks[0]]
    for row, col in colored_blocks:
        rmemory[row].append(col)
        cmemory[col].append(row)
        y[row, min(rmemory[row]):max(rmemory[row]) + 1] = color
        y[min(cmemory[col]):max(cmemory[col]) + 1, col] = color
    return y


def solve_b775ac94(x):
    """
    """
    def find_structures(grid):
        """
        Find connected structures based on their colors and return their coordinates in the grid.
        """
        labeled = skimage.measure.label(grid, connectivity=2)
        objects = (np.where(labeled == label) for label in np.unique(labeled) if label)
        for obj in objects:
            yield np.array(obj).T

    def find_closest_expandable_roots(structure, expandable_roots, distance_threshold=2):
        """
        Return the expandable roots closest to the structure.
        """
        kdtree = scipy.spatial.KDTree(structure)
        for root in expandable_roots:
            distance, index = kdtree.query(root)
            if distance <= distance_threshold:
                yield (structure[index], root)
                
    def mirror_structure(structure, structure_root, expandable_root):
        """
        Calculate the mirror coordinates of a structure based on the position of a expandable root
        in relation to the structure's root.
        """
        x1, y1 = structure_root[0][0], structure_root[0][1]    
        x2, y2 = expandable_root[0][0], expandable_root[0][1]
        blueprint = structure_root - structure

        if x2 == x1 and y2 != y1:
            return blueprint @ [[-1, 0], [0, 1]] + expandable_root
        if x2 != x1 and y2 == y1:
            return blueprint @ [[1, 0], [0, -1]] + expandable_root
        if x2 != x1 and y2 != y1:
            return blueprint @ [[1, 0], [0, 1]] + expandable_root
        else:
            return blueprint @ [[0, 1], [1, 0]] + expandable_root
                
    def expand_structures(grid):
        """
        Expand roots closest to the large structures found in the grid.
        """
        structures = list(find_structures(grid))
        large_structures = filter(lambda structure: len(structure) > 1, structures)
        expandable_roots = list(filter(lambda structure: len(structure) == 1, structures))
        for structure in large_structures:
            structure_expandable_roots = find_closest_expandable_roots(structure, expandable_roots)
            for structure_root, expandable_root in structure_expandable_roots:
                color = grid[tuple(expandable_root.T)][0]
                mirrored_structure = mirror_structure(structure, structure_root, expandable_root)
                grid[tuple(mirrored_structure.T)] = color
        return grid

    y = expand_structures(np.copy(x))
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
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
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
