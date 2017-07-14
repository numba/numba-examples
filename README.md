# Numba Examples

This repository contains examples of using [Numba](https://numba.pydata.org)
to implement various algorithms.  If you want to browse the examples and
performance results, head over to the [examples site](https://numba.pydata.org/numba-examples/index.html).

In the repository is a benchmark runner (called `numba_bench`) that walks a directory tree of benchmarks, executes them, saves the results in JSON format, then generates HTML pages with pretty-printed source code and performance plots.

We are actively seeking new Numba examples!  Keep reading to learn how benchmarks are defined.

## Making a benchmark

A *benchmark* is directory containing at least two files:

  * A `bench.yaml` describing the benchmark and how to run it.
  * One or more Python files that contain the benchmark functions.

A typical `bench.yaml` looks like this:

``` yaml
name: Zero Suppression
description: |
    Map all samples of a waveform below a certain absolute magnitude to zero
input_generator: impl.py:input_generator
xlabel: Number of elements
validator: impl.py:validator
implementations:
    - name: numpy
      description: Basic NumPy implementation
      function: impl.py:numpy_zero_suppression
    - name: numba_single_thread_ufunc
      description: Numba single threaded ufunc
      function: impl.py:numba_zero_suppression
    - name: numba_gpu_ufunc
      description: |
          Numba GPU ufunc.  Note this will be slower than CPU!
          There is not enough work for the GPU to do, so the fixed overhead dominates.
      function: gpu.py:numba_zero_suppression
      requires:
          - gpu
baseline: numpy
```

The top-level keys are:

  * `name`: A short name for the example.  Used in plot titles, page titles, etc.  Keep it short.
  * `description`: A Markdown description of the example.  This can be multiple lines and is put at the top of the example page.
  * `input_generator`: Python function to call to generate input data.  Format is `filename:function_name`.
  * `validator`: Python function to verify that output is correct.  (You don't want to benchmark a function that gives wrong answers!)
  * `implementations`: A list of implementations to test.  Being able to compare multiple implementations is important to see whether Numba is providing any benefit.  Different implementations also have different scaling characteristics, which is helpful to compare.
  * `baseline`: The name of the implementation to use as the "reference" when computing speedup ratios for the other implementations.
  
Each implementation also defines:

  * `name`: Short name of implementation.  Used in legends, tabs, and other places.
  * `description`: Longer Markdown description of implementation.  Can be multi-line.
  * `function`: Python function with implementation.  Note that multiple implementations can be in the same file, or they can be in different files.
  * `requires`: A list of strings indicating resources that are required to run this benchmark.  If the benchmark runner is not told (with a command line option) that it has the required resources for an implementation, that implementation will be skipped.
  
### Benchmarking Process

When benchmarking an example, the runner does the following:

  1. All of the functions are loaded into memory by calling `execfile` on all of the Python scripts mentioned in `bench.yaml`.  No file is loaded more than once, even if multiple implementations refer to it.
  2. The input generator is called and for each input set it yields and each implementation that is defined:
    a. The implementation is called once, and the output (along with the input) sent to the validator function to be checked.  This also triggers any JIT compilation in the implementation so it does not contribute to the time measurement.
    b. The implementation is called many times with the same input in a loop to get a more accurate time measurement, using roughly the same automatic scaling logic as `%timeit` in Jupyter so each batch of calls takes between 0.2 and 2 seconds.  The best of three batches is recorded.
  3. Results are grouped by category (see description of input generator below) so that one plot will be made for each unique category and each plot will contain one series per implementation.
  
### Input Generators

An input generator is a Python generator that yields dictionaries each containing an input data set to benchmark.  An example looks like:
``` python
def input_generator():
    for dtype in [np.int16, np.float32, np.float64]:
        for size in [100, 1000, 10000, 50000]:
            name = np.dtype(dtype).name
            input_array = np.random.normal(loc=0.0, scale=5.0, size=size)
            # add a pulse train
            input_array += 50 * np.clip(np.cos(np.linspace(0.0, 1.0, num=size)*np.pi*10), 0, 1.0)
            input_array = input_array.astype(dtype)
            yield dict(category=('%s' % name,), x=size, input_args=(input_array, 8.0), input_kwargs={})
```
Each dictionary has the following keys:

  * `category`: A tuple of strings that can be used to create different plots for the same example.  In the above case, the category is used to indicate the data type of the array, so that a separate plot will be made for `int16`, `float32`, and `float64`.  This could be used to group inputs into different categories like `square array`, `tall and skinny`, etc.
  * `x`: A float or int that denotes the input size.  The meaning of this value is entirely up to the example author.  It will be used as the x-axis in the performance plots.  Usually number of array elements is a good choice, but it could be some other size metric.
  * `input_args`: A tuple of positional arguments to the implementation function
  * `input_kwargs`: A dictionary of keyword arguments to the implementation function
  
### Validator

A validator function takes one set of input args and kwargs yielded by the input generator, and the output from the execution of one of the implementations, and determines if that output is correct.  An example looks like:
``` python
def validator(input_args, input_kwargs, impl_output):
    # We're using the Numpy implementation as the reference
    expected = numpy_zero_suppression(*input_args, **input_kwargs)
    np.testing.assert_array_equal(expected, impl_output)
```
As the comment notes, we are treating the NumPy implementation as the reference, but validation can be done any way that makes sense.  If the output is incorrect, an `AssertionError` should be raised.

### Marking Implementation Source Code

The output HTML from running the benchmark includes the source code of the implementation.  Since the implementation might depend on imports and helper functions, by default the benchmark runner will snapshot *the entire Python file* containing the main implementation function for the HTML output.

For short benchmarks, it might be more convenient to put more than one implementation into the same file.  In that case, special comments can be used to tell the runner what section of code to capture.  For example, in this file:
``` python
import numpy as np

def input_generator():
    for dtype in [np.int16, np.float32, np.float64]:
        for size in [100, 1000, 10000, 50000]:
            name = np.dtype(dtype).name
            input_array = np.random.normal(loc=0.0, scale=5.0, size=size)
            # add a pulse train
            input_array += 50 * np.clip(np.cos(np.linspace(0.0, 1.0, num=size)*np.pi*10), 0, 1.0)
            input_array = input_array.astype(dtype)
            yield dict(category=('%s' % name,), x=size, input_args=(input_array, 8.0), input_kwargs={})

#### BEGIN: numpy
import numpy as np

def numpy_zero_suppression(values, threshold):
    result = np.zeros_like(values)
    selector = np.abs(values) >= threshold
    result[selector] = values[selector]
    return result
#### END: numpy
```
The implementation named `numpy` will only show the code between `#### BEGIN: numpy` and `#### END: numpy` in the HTML rendering of the example.

## Running Benchmarks

Assuming you have this repository checked out into the current directory, you should do the following to setup your environment to run the benchmarks:
```
conda create -n numba_bench --file conda-requirements.txt
source activate numba_bench
python setup.py install
```

The most common way to run the benchmarks is like this:
```
numba_bench -o results
```
or
```
conda install cudatoolkit # required for Numba GPU support
numba_bench -o results -r gpu
```
to run all the benchmarks, including the "gpu" only benchmarks.  The `results/` directory will contain an `index.html` with a list of the examples that were run, and each subdirectory will contain a `results.json` and `results.html` file containing the raw performance data and generated plot HTML, respectively.

There are additional options to `numba_bench` that can be useful:

  * `--skip-existing`: Skip any example with a `results.json` already present in the output directory.
  * `--run-only`: Only run benchmarks, don't generate HTML output.
  * `--plot-only`: Only generate HTML output, don't run benchmarks.
  * `--root`: Set the root of the tree of benchmarks.  By default it is the current directory.
  
In addition, substrings can be listed on the command line that will limit which tests will run.  For example, this command:
```
numba_bench -o results waveform pdf
```
will run any test under the benchmark tree with a directory that contains `waveform` or `pdf` in the name.
