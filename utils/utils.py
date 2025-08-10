import argparse


def parse_tuple(string):
    try:
        items = string.strip("()").split(",")
        return tuple(int(i) for i in items)
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be numbers separated by commas")