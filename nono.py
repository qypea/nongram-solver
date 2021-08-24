#!/usr/bin/env python3

import enum
import numpy
import unittest


class Square(enum.IntEnum):
    """Enum for the state of a single square."""
    UNKNOWN = 0
    EMPTY = 1
    FULL = 2


class Nonogram:
    """Collection of the data required to represent a nonogram."""
    def __init__(self, row_size, col_size, row_inputs, col_inputs):
        assert row_size == len(row_inputs)
        assert col_size == len(col_inputs)
        self.array = numpy.ndarray([row_size, col_size],
                                   dtype=numpy.ubyte,
                                   order='C')
        self.array.fill(Square.UNKNOWN)

        self.row_inputs = row_inputs
        self.col_inputs = col_inputs
        self._row_labels = []
        self._row_pad = ""
        self._col_labels = []
        self._max_col_labels = 0

        self._gen_labels()

    def _gen_labels(self):
        """Generate the row and column labels for printing."""
        # Build up the row labels
        for row_input in self.row_inputs:
            label = ""
            for i in row_input:
                label += "{0: <2} ".format(i)
            self._row_labels.append(label)

        max_label = max([len(x) for x in self._row_labels])
        for i, label in zip(range(len(self._row_labels)), self._row_labels):
            while len(label) < max_label:
                label = " " + label
            self._row_labels[i] = label

        self._row_pad = " " * max_label

        # Build up column labels
        self._max_col_labels = max([len(label) for label in self.col_inputs])
        for column in self.col_inputs:
            self._col_labels.append(["{0: <2} ".format(label) for label in column])

        for label in self._col_labels:
            while len(label) < self._max_col_labels:
                label.insert(0, "   ")

    def __str__(self):
        """Print our state prettily."""
        ret = ""

        # Columns
        for i in range(self._max_col_labels):
            ret += self._row_pad
            for label in self._col_labels:
                ret += label[i]
            ret += "\n"

        # Actual data rows
        for row_label, row in zip(self._row_labels, self.array):
            ret += row_label
            for item in row:
                if item == Square.UNKNOWN:
                    ret += " "
                elif item == Square.EMPTY:
                    ret += "X"
                elif item == Square.FULL:
                    ret += "#"
                else:
                    raise InvalidValue("Unexpected square value to print")
                ret += "  "
            ret += "\n"

        return ret

    def row(self, i):
        """Return the row i."""
        return self.row_inputs[i], self.array[i]

    def column(self, i):
        """Return the column i."""
        return self.col_inputs[i], self.array[:,i]


def parse_row(state: list) -> list:
    """Given a state extract the inputs for that state."""
    assert Square.UNKNOWN not in state, "No empty squares allowed"

    if len(state) == 0:
        return []

    try:
        start = list(state).index(Square.FULL)
    except ValueError:
        return []

    try:
        end = list(state).index(Square.EMPTY, start)

        length = len(state[start:end])
        rest = state[end:]
    except ValueError:
        length = len(state[start:])
        rest = []

    substate = parse_row(rest)

    return [length] + substate


class TestParseRow(unittest.TestCase):
    """Test parse_row function."""
    def test_parse_row(self):
        """Test parse_row function."""
        cases = [([], []),
                 ([1], []),
                 ([2], [1]),

                 ([1, 1], []),
                 ([2, 1], [1]),
                 ([1, 2], [1]),
                 ([2, 2], [2]),

                 ([1, 1, 1], []),
                 ([2, 1, 1], [1]),
                 ([1, 2, 1], [1]),
                 ([2, 2, 1], [2]),
                 ([1, 1, 2], [1]),
                 ([2, 1, 2], [1, 1]),
                 ([1, 2, 2], [2]),
                 ([2, 2, 2], [3]),

                 ([1, 1, 1, 1], []),
                 ([2, 1, 1, 1], [1]),
                 ([1, 2, 1, 1], [1]),
                 ([2, 2, 1, 1], [2]),
                 ([1, 1, 2, 1], [1]),
                 ([2, 1, 2, 1], [1, 1]),
                 ([1, 2, 2, 1], [2]),
                 ([2, 2, 2, 1], [3]),
                 ([1, 1, 1, 2], [1]),
                 ([2, 1, 1, 2], [1, 1]),
                 ([1, 2, 1, 2], [1, 1]),
                 ([2, 2, 1, 2], [2, 1]),
                 ([1, 1, 2, 2], [2]),
                 ([2, 1, 2, 2], [1, 2]),
                 ([1, 2, 2, 2], [3]),
                 ([2, 2, 2, 2], [4]),
                ]
        for state, expected_input in cases:
            with self.subTest(state=state):
                self.assertEqual(parse_row(state), expected_input)


def check_row(inputs, state):
    """Check if a row is valid."""
    actuals = parse_row(state)
    return inputs == actuals


def generate_row_options(state: list):
    """Generate all possible permutations of a row given the current state."""
    state = state.copy()
    try:
        i = list(state).index(Square.UNKNOWN)
    except ValueError:
        yield state
        return

    state[i] = Square.EMPTY
    for derived in generate_row_options(state):
        yield derived
    state[i] = Square.FULL
    for derived in generate_row_options(state):
        yield derived
    state[i] = Square.UNKNOWN


class TestGenerateRow(unittest.TestCase):
    """Test the generate_row_options function."""
    def test_generate_row_options(self):
        """Test the generate_row_options function."""
        cases = [
            ([], [[]]),
            ([0], [[1], [2]]),
            ([1], [[1]]),
            ([2], [[2]]),
            ([0, 0], [[1, 1], [1, 2], [2, 1], [2, 2]]),
            ([1, 0], [[1, 1], [1, 2]]),
            ([0, 1], [[1, 1], [2, 1]]),
            ]
        for state, expected in cases:
            with self.subTest(state=state):
                expected.sort()
                actual = list(generate_row_options(state))
                actual.sort()
                self.assertEqual(actual, expected)


def row_probabilities(inputs, state):
    """Calculate the probabilities of each square being filled in."""
    counts = [0.0] * len(state)
    valid = 0

    for possible_state in generate_row_options(state):
        if not check_row(inputs, possible_state):
            continue

        valid += 1
        for i in range(len(state)):
            if possible_state[i] == Square.FULL:
                counts[i] += 1.0

    return [x / valid for x in counts]


class TestRowProb(unittest.TestCase):
    """Test row probabilities function."""
    def test_row_probabilities(self):
        """Test row probabilities function."""
        cases = [
            ([], [0, 0, 0], [0, 0, 0]),
            ([1], [0, 0, 0], [1.0/3, 1.0/3, 1.0/3]),
            ([2], [0, 0, 0], [1.0/2, 1, 1.0/2]),
            ([1, 1], [2, 0, 0], [1, 0, 1]),
            ]
        for inputs, state, expected in cases:
            with self.subTest(inputs=inputs, state=state):
                self.assertEqual(row_probabilities(inputs, state), expected)


# Fill in 0s, 1s with x, #
# Check if something changed in filling in process
# Convert columns to rows, fill those in too
# Loop through all rows, columns filling in things. Print state after each change


# Move to checking row+column together to get more data out
# Guess and check recursive solution when we run out of simple steps

if __name__ == "__main__":
    unittest.main()
