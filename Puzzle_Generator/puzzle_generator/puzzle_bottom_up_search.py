from puzzle_piece import PuzzlePiece
from puzzle_piece import Rotation
from puzzle_piece import PieceSide
from puzzle import Puzzle
from random import shuffle

def perform_bottom_up_search(puzzle):

    # Get the piece breakdown information
    x_count = puzzle._x_piece_count
    y_count = puzzle._y_piece_count
    grid_length = max(x_count, y_count)

    # Build an array that is larger than the puzzle as it may build in any direction around the board
    # noinspection PyUnusedLocal
    solution_grid = [[None for y in range(0, 2 * grid_length + 1)] for x in range(0, 2 * grid_length + 1)]
    # Initialize the board information
    top__left = bottom_right = center = (grid_length, grid_length)


    # Get the puzzle's pieces and transfer them to a frontier set.
    pieces = puzzle.pieces()
    unexplored_set = [pieces[x][y] for x in range(0, x_count) for y in range(0, y_count)]
    shuffle(unexplored_set)

    # Select the first piece of the puzzle.
    first_piece = unexplored_set.pop()
    frontier_set = {}
    frontier_set.update(center, first_piece)
    solution_grid[center[0]][center[1]] = first_piece

    # Iterate until all pieces have been explored.
    while len(unexplored_set) > 0:

        # Get the next piece to assign.
        next_piece = select_next_piece(unexplored_set, frontier_set, top_left, bottom_right, x_count, y_count)

        # Remove the piece from the unexplored set and place it in the board
        unexplored_set.remove(next_piece)
        next_piece_coord = next_piece.assigned_location
        solution_grid[next_piece_coord[0]][next_piece_coord[1]] = next_piece

        # Assume piece is in the frontier. Checked below.
        frontier_set.update(next_piece_coord, next_piece)

        # Update the board edge coordinates.
        top_left = (min(next_piece_coord[0], top_left[0]), min(next_piece_coord[1], top_left[1]))
        bottom_right = (max(next_piece_coord[0], bottom_right[0]), max(next_piece_coord[1], bottom_right[1]))

        # Check the pieces neighbors and see if they can be removed from the frontier set.
        # If they are and have no available neighbors, then remove the piece from the frontier.
        neighbor_coords = [next_piece_coord,
                           (next_piece_coord[0] - 1, next_piece_coord[1]),  # Left Neighbor
                           (next_piece_coord[0] + 1, next_piece_coord[1]),  # Right Neighbor
                           (next_piece_coord[0], next_piece_coord[1] - 1),  # Top neighbor
                           (next_piece_coord[0], next_piece_coord[1] + 1)]  # Bottom Neighbor
        for coord in neighbor_coords:
            neighbor = frontier_set.get(coord)
            if neighbor is not None:
                available_neighbors = determine_available_neighbors(coord, solution_grid,
                                                                    top_left, bottom_right, x_count, y_count)
                # If it has no neighbors delete from the frontier
                if(len(available_neighbors) == 0):
                    frontier_set.pop(coord)

    # Make the puzzle_solution
    solution_puzzle = Puzzle()
    #solution_puzzle = make_puzzle_solution(solution_grid, top_left, bottom_right, x_count, y_count)
    solution_puzzle.export_puzzle("solution.bmp")

def determine_available_neighbors(piece_coord, solution_grid, top_left, bottom_right, x_count, y_count):
    pass

def select_next_piece(solution_grid, unexplored_set, frontier_set, top_left, bottom_right, x_count, y_count):

    min_distance = sys.maxint
    best_piece = None
    best_piece_coord = None
    best_piece_rotation = None

    for frontier_coord in frontier_set.keys():
        available_neighbors = determine_available_neighbors(frontier_coord, solution_grid, top_left, bottom_right,
                                                            x_count, y_count)
        # in some rare cases (e.g. edge of board reached, a frontier piece may have no neighbors. If so continue.
        if(len(available_neighbors) == 0): continue

        frontier_piece = frontier_set.get(frontier_coord)

        # Iterate through the unexplored pieces.
        for new_piece in unexplored_set:
            for rotation in Rotation.get_all_rotations():
                # Set the rotation of the piece.
                new_piece.rotation = rotation

                # Go through all available edges for the frontier pin.
                for frontier_edge in available_neighbors:
                    other_edge = frontier_edge.get_paired_edge()

                    # Calculate the inter-piece distance.
                    piece_distance = PuzzlePiece.get(frontier_piece, frontier_edge, new_piece, other_edge)
                    if best_piece is None or piece_distance < min_distance:
                        best_piece = new_piece
                        best_piece_coord = frontier_piece.get_neighbor_coordinate(frontier_edge)
                        best_piece_rotation = rotation
                        min_distance = piece_distance

    # Set the best piece's assigned location and rotation
    best_piece.assigned_location = best_piece_coord
    best_piece.rotation = best_piece_rotation
    # Return the best piece
    return best_piece

def make_puzzle_solution(solution_grid, top_left, bottom_right, x_count, y_count):
    # Build the output_grid
    final_grid = [[None in y in range(0, y_count)] in x in range(0, x_count)]
    # Check if the board is unrotated
    if(top_left[0] - bottom_right[0] + 1 == x_count):
        for x in range(0, x_count):
            for y in range(0, y_count):
                final_grid[x][y] = solution_grid[top_left[0] + x][top_left[1] + y]
    # Board is rotated
    else:
        # Need to swap x and y axis in solution since rotation
        for x in range(0, x_count):
            for y in range(0, y_count):
                final_grid[x][y] = solution_grid[bottom_right[0] - y][top_left[1] + x]

    # Return a puzzle built from the individual pieces.
    return Puzzle.make_puzzle_from_pieces(final_grid)

if __name__ == '__main__':
    puzzles = [("duck.bmp", (10, 10)), ("two_faced_cat.jpg", (20, 10))]
    for puzzle_info in puzzles:
        # Extract the information on the images
        file = puzzle_info[0]
        (x_count, y_count) = puzzle_info[1]
        # Build a test puzzle
        test_puzzle = Puzzle(Puzzle.DEFAULT_IMAGE_PATH + file)
        # test_puzzle.set_puzzle_image(Puzzle.DEFAULT_IMAGE_PATH + file )
        # test_puzzle.open_image()
        test_puzzle.convert_to_pieces(x_count, y_count)
        test_puzzle.shuffle_pieces()
        test_puzzle.export_puzzle(Puzzle.DEFAULT_IMAGE_PATH + "puzzle_" + file)