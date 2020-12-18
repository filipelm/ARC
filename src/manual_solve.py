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
    The task starts with blocks spread throughout the grid. The objective
    is to connecte those blocks by drawing straight lines between them.
    
    To solve the task, this function will identify the position of the existing blocks
    in the grid, then for each block found it will look for other blocks in the same 
    line or column and connect them.
    
    This solution solves all train and test inputs correctly.
    """
    y = np.copy(x)
    rmemory, cmemory = defaultdict(list), defaultdict(list)
    # Find existing blocks.
    colored_blocks = list(zip(*np.where(x)))
    # Get the color of existing blocks. They all have the same color.
    color = x[colored_blocks[0]]
    for row, col in colored_blocks:
        # Keep track of the block's row.
        rmemory[row].append(col)
        # Keep track of the block's column.
        cmemory[col].append(row)
        # Draw a straight line between all the blocks that appear in the row.
        y[row, min(rmemory[row]):max(rmemory[row]) + 1] = color
        # Draw a straight line between all the blocks that appeat in the column.
        y[min(cmemory[col]):max(cmemory[col]) + 1, col] = color
    return y


def solve_b775ac94(x):
    """
    The task starts with some objects (structures) with different colors sitting around the grid.
    The objective is to mirror the largest part of each object (structure) based on the position
    of smaller adjacent objects. Each object mirror should have the same color of the adjacent object 
    (expandable root).

    To the solve the task, this function uses skimage.measure.label to find the connected components
    in the grid based on their colors and position. That will split the large objects (structures) from the smaller
    ones (expandable roots). Then for each expandable root it will look for the closest large structure
    and calculate the position of the structure mirror relative to the expandable root.

    This solution solves all train and test inputs correctly.
    """
    def find_connected_components(grid):
        """
        Return the coordinates of connected components in the grid.
        """
        # Label the connected components in the grid.
        labeled = skimage.measure.label(grid, connectivity=2)
        # Extract the coordinate of each structure and expandable root
        # based on the connected compoenents labels.
        components = (np.where(labeled == label) for label in np.unique(labeled) if label)
        for obj in components:
            yield np.array(obj).T

    def find_closest_expandable_roots(structure, expandable_roots, distance_threshold=2):
        """
        Return the expandable roots closest to the structure.
        """
        # Index the coordinates that makes up a structure using
        # scipy.spatial.KDTree to make it easy to query which
        # expandable roots are closer to it.
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

        # The structure root is the block that is connected to all the
        # expandable roots. Here we calculate the coordinates of the 
        # structure relative to it's root. We can read the resulting list of coordinates as: 
        # there is one block to right of the root, one block below the root, and so on.
        blueprint = structure_root - structure

        # Then we rotate the blueprint based on the direction of the
        # expandable root. We sum the position of the expandable root
        # to the rotated structure to get the coordinates of the
        # new structure.
        if x2 == x1 and y2 != y1:
            return blueprint @ [[-1, 0], [0, 1]] + expandable_root
        if x2 != x1 and y2 == y1:
            return blueprint @ [[1, 0], [0, -1]] + expandable_root
        if x2 != x1 and y2 != y1:
            return blueprint @ [[1, 0], [0, 1]] + expandable_root
                
    def expand_structures(grid):
        """
        Expand roots closest to the large structures found in the grid.
        """
        structures = list(find_connected_components(grid))
        large_structures = filter(lambda structure: len(structure) > 1, structures)
        expandable_roots = list(filter(lambda structure: len(structure) == 1, structures))
        for structure in large_structures:
            roots = find_closest_expandable_roots(structure, expandable_roots)
            for structure_root, expandable_root in roots:
                # Finds the color of the expandable root.
                color = grid[tuple(expandable_root.T)][0]
                # Calculate mirror relative to the expandable root.
                mirrored_structure = mirror_structure(structure, structure_root, expandable_root)
                # Assign expandable root color to all coordinates that makes up
                # the new mirrored structure. 
                grid[tuple(mirrored_structure.T)] = color
        return grid

    y = expand_structures(np.copy(x))
    return y


def solve_c8cbb738(x):
    """
    The task starts with objects of different colors sitting around the grid.
    The objective is create a new grid where all the objects fit each other.

    To solve the task, this function isolate each object on it's own grid. Then
    the largest object grid is identified and all grids are padded to the size
    of the largest object grid. Finnaly it will sum all object grids to find
    the resulting grid where all objects fit.

    This solution solves all train and test inputs correctly.
    """
    def dominant_color(grid):
        """
        Find the most common color in the grid.
        """
        return np.argmax(np.bincount(grid.flat))

    def find_objects_colors(grid, background):
        """
        Find the color of objects in the grid that are different
        from the grid's background.
        """
        return np.unique(grid[np.where(grid != background)])

    def isolate(grid, color):
        """
        Query object coordinates by color, then create a grid that
        contains only the object found with the color.
        """
        # Find the object based on it's color and draw it again
        # in a new grid with the same shape as the original grid.
        object_grid = np.zeros_like(grid)
        object_grid[np.where(grid == color)] = color
        # Reduce the new grid to a smaller grid that contains only
        # the identified object.
        x, y = np.nonzero(object_grid)
        return object_grid[x.min():x.max()+1, y.min():y.max()+1]

    def reshape(grid, shape):
        """
        Reshape a object grid to the specified shape. The object
        will be centered in the resulting grid.
        """
        y_target, x_target = shape
        y_origin, x_origin = grid.shape
        # calculate the size of row and column pad based on the given shape.
        x, y = x_target-x_origin, y_target-y_origin
        # Pad the grid by adding the missing rows and columns.
        return np.pad(grid, [(y//2, y//2 + y%2), (x//2, x//2 + x%2)])

    background = dominant_color(x)
    objects = [isolate(x, color) for color in find_objects_colors(x, background)]
    largest_object = max(objects, key=np.size)
    # Sum all objects grids given that they all have the same shape and are centered
    # on their respective grids.
    y = sum(map(lambda object: reshape(object, largest_object.shape), objects))
    # Assign the background color to the black blocks in the resulting grid.
    y[np.where(y == 0)] = background
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
