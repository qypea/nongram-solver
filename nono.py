#!/usr/bin/env python3

import unittest

def parse_row(state: str) -> list:
    """Given a state extract the inputs for that state."""
    assert " " not in state, "No empty squares allowed"

    if state == "":
        return []

    start = state.find("#")
    if start == -1:
        return []

    end = state.find("x", start)
    if end == -1:
        length = len(state[start:])
        rest = ""
    else:
        length = len(state[start:end])
        rest = state[end:]

    substate = parse_row(rest)

    return [length] + substate


class TestParseRow(unittest.TestCase):
    """Test parse_row function."""
    def test_parse_row(self):
        """Test parse_row function."""
        cases = [("", []),
                 ("x", []),
                 ("#", [1]),

                 ("xx", []),
                 ("#x", [1]),
                 ("x#", [1]),
                 ("##", [2]),

                 ("xxx", []),
                 ("#xx", [1]),
                 ("x#x", [1]),
                 ("##x", [2]),
                 ("xx#", [1]),
                 ("#x#", [1, 1]),
                 ("x##", [2]),
                 ("###", [3]),

                 ("xxxx", []),
                 ("#xxx", [1]),
                 ("x#xx", [1]),
                 ("##xx", [2]),
                 ("xx#x", [1]),
                 ("#x#x", [1, 1]),
                 ("x##x", [2]),
                 ("###x", [3]),
                 ("xxx#", [1]),
                 ("#xx#", [1, 1]),
                 ("x#x#", [1, 1]),
                 ("##x#", [2, 1]),
                 ("xx##", [2]),
                 ("#x##", [1, 2]),
                 ("x###", [3]),
                 ("####", [4]),
                ]
        for state, expected_input in cases:
            with self.subTest(state=state):
                self.assertEqual(parse_row(state), expected_input)


def check_row(inputs, state):
    """Check if a row is valid."""
    actuals = parse_row(state)
    return inputs == actuals


# Generate all possible permutations of a row given the current state
# Filter with check_row so we only have the valid ones
# Compute probabilities
# Fill in 0s, 1s with x, #
# Check if something changed in filling in process
# Convert columns to rows, fill those in too
# Loop through all rows, columns filling in things. Print state after each change


# Move to checking row+column together to get more data out
# Guess and check recursive solution when we run out of simple steps

if __name__ == "__main__":
    unittest.main()
