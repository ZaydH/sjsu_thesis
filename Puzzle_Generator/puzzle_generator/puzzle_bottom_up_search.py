from puzzle_piece import PuzzlePiece
from puzzle import Puzzle
from random import shuffle

def perform_bottom_up_search(puzzle):


    grid_length = max(puzzle._x_piece_count, puzzle._y_piece_count)

    # Build an array that is larger than the puzzle as it may build in any direction around the board
    # noinspection PyUnusedLocal
    solution_grid = [[None for y in range(0, 2 * grid_length + 1)] for x in range(0, 2 * grid_length + 1)]
    # Initialize the board information
    top__left = bottom_right = center = (grid_length, grid_length)


    # Get the puzzle's pieces and transfer them to a frontier set.
    pieces = puzzle.get_pieces()
    unexplored_set = [pieces[x][y] for x in range(0, x_count) for y in range(0, y_count)]
    shuffle(unexplored_set)

    # Select the first piece of the puzzle.
    first_piece = unexplored_set.pop()
    frontier_set = {}
    frontier_set.update(center, first_piece)
    solution_grid[center[0]][center[1]] = first_piece

    # Iterate until all pieces have been explored.
    while len(unexplored_set) > 0:
        (next_piece, coordinate, rotation) = select_next_piece(unexplored_set, frontier_set, top_left, bottom_right,
                                                               x_count, y_count)

        # Remove the piece from the unexplored set and place it in the board
        unexplored_set.remove(next_piece)
        solution_grid[coordinate[0]][coordinate[1]] = next_piece
        frontier_set.update(coordinate, next_piece) # Assume piece is in the frontier. Checked below.

        # Update the edges of the board.
        top_left = (min(coordinate[0], top_left[0]), min(coordinate[1], top_left[1]))
        bottom_right = (max(coordinate[0], bottom_right[0]), max(coordinate[1], bottom_right[1]))

        # Check the pieces neighbors and see if they can be removed from the frontier set.
        # If they are and have no available neighbors, then remove it.
        neighbor_coords = [coordinate,
                           (coordinate[0] - 1, coordinate[1]), (coordinate[0] + 1, coordinate[1]),  # X Neighbors
                           (coordinate[0], coordinate[1] - 1), (coordinate[0], coordinate[1] + 1)]  # Y Neighbors
        for coord in neighbor_coords:
            neighbor = frontier_set.get(coord)
            if neighbor is not None:
                available_neighbors = determine_available_neighbors(coordinate, solution_grid,
                                                                    top_left, bottom_right, x_count, y_count)
                # If it has available neighbors add it to the frontier.
                if(len(available_neighbors) == 0):
                    frontier_set.update(coordinate, next_piece)

    # Make the puzzle_solution
    solution_puzzle = Puzzle()
    #solution_puzzle = make_puzzle_solution(solution_grid, top_left, bottom_right, x_count, y_count)
    solution_puzzle.export_puzzle("solution.bmp")

def determine_available_neighbors(piece_coord, solution_grid, top_left, bottom_right, x_count, y_count):
    pass

def select_next_piece(solution_grid, unexplored_set, frontier_set, top_left, bottom_right, x_count, y_count):

    for frontier_coord in frontier_set.keys():
        available_neighbors = determine_available_neighbors(frontier_coord, solution_grid, top_left, bottom_right,
                                                            x_count, y_count)
        # in some rare cases (e.g. edge of board reached, a frontier piece may have no neighbors. If so continue.
        if(len(available_neighbors) == 0): continue

        frontier_piece = frontier_set.get(frontier_coord)

        # Iterate through the unexplored pieces.
        for new_piece in unexplored_set:
            for rotation in Rotation.get_all_rotations():
                # Set the rotation of a piece.
                new_piece.set_rotation(new_piece)
                for neighbor_edge in available_neighbors:
                    # Calculate the inter-piece distance.
                    piece_distance = PuzzlePiece.get(frontier_piece, , new_piece, )

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