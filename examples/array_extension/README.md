# An array extension example

Warning: This example is intended for early adopters. 
         It currently uses and duplicates internal Numba code to achieve its goal. 

## How to use

Use the `Makefile` in tutorial to build and test.

The source code (`extarray.py` and `extarray.c`) contains details comments.
`extarray.py` is the main file. It is divided into several parts.
Each explaining a step in making a user-defined array that reuses internal
Numba implementation for the NumPy array support.

## Enviornment requirements

- Numba installed. Can use `pip install -e ..`.
- `pytest`

