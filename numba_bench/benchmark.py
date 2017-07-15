#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import time
from collections import defaultdict
import datetime
import argparse
import json
import re

import yaml


if sys.version_info >= (3,3):
    CLOCK_FUNCTION = time.perf_counter
else:
    CLOCK_FUNCTION = time.clock


if sys.version_info <= (3, 0):
    range = xrange


BENCH_CONFIG_FILENAME = 'bench.yaml'


class BenchmarkError(Exception):
    def __init__(self, benchmark_dir, message):
        self.benchmark_dir = benchmark_dir
        self.message = message

    def __str__(self):
        return '[%s]: %s' % (self.benchmark_dir, self.message)


class Benchmark(object):
    @staticmethod
    def is_benchmark_dir(dirname):
        if os.path.isdir(dirname):
            if os.path.isfile(os.path.join(dirname, BENCH_CONFIG_FILENAME)):
                return True

        return False

    def __init__(self, benchmark_dir, resources):
        self.benchmark_dir = benchmark_dir
        self.benchmark_config_filename = os.path.join(benchmark_dir, BENCH_CONFIG_FILENAME)
        self.python_file_cache = {}

        with open(self.benchmark_config_filename, 'r') as f:
            config = yaml.load(f)

        self._validate_and_normalize_config(config, resources)

    def _raise_benchmark_error(self, message):
        raise BenchmarkError(self.benchmark_dir, message)

    def _load_function(self, descriptor):
        filename, function_name = descriptor.split(':')
        path = os.path.join(self.benchmark_dir, filename)

        if path not in self.python_file_cache:
            # Load Python file
            global_dict = {}

            with open(path) as f:
                code = compile(f.read(), path, 'exec')
                exec(code, global_dict)

            self.python_file_cache[path] = global_dict

        try:
            return self.python_file_cache[path][function_name]
        except KeyError:
            self._raise_benchmark_error('Unable to find function "%s" in %s' % (function_name, filename))

    def _load_code_fragment(self, descriptor, name):
        filename, function_name = descriptor.split(':')
        path = os.path.join(self.benchmark_dir, filename)
        with open(path, 'r') as f:
            contents = f.read()

        # Find code section
        pattern = r'#### BEGIN: %s$(.*)^#### END: %s$' % (name, name)
        match = re.search(pattern, contents, re.DOTALL | re.MULTILINE)
        if match is None:
            return contents
        else:
            return match.group(1)

    def _validate_and_normalize_impl(self, impl, resources):
        try:
            name = impl['name']
        except KeyError:
            self._raise_benchmark_error('Benchmark implementation missing "name" attribute')

        # optional set of requirement flags
        requires = set(impl.get('requires', []))
        if not requires.issubset(resources):
            return None

        # optional description
        description = impl.get('description')

        try:
            function_descriptor = impl['function']
        except KeyError:
            self._raise_benchmark_error('Benchmark implementation %s missing "function" attribute' % name)

        function = self._load_function(function_descriptor)

        source = self._load_code_fragment(function_descriptor, name)

        return dict(name=name, description=description, function=function, source=source, requires=requires)

    def _validate_and_normalize_config(self, config, resources):
        # Benchmark name
        try:
            self.name = config['name']
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "name" attribute')

        # Benchmark description (optional)
        self.description = config.get('description', '')

        # Input generator function
        try:
            input_generator_descriptor = config['input_generator']
            self.input_generator = self._load_function(input_generator_descriptor)
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "input_generator" attribute')

        # Validator function
        try:
            validator_descriptor = config['validator']
            self.validator = self._load_function(validator_descriptor)
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "validator" attribute')
        
        # X-axis label
        try:
            self.xlabel = config['xlabel']
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "xlabel" attribute')

        # baseline implementation name
        try:
            self.baseline_name = config['baseline']
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "baseline" attribute')

        # Load implementations
        try:
            all_implementations = config['implementations']
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "implementations" attribute')

        self.implementations = []
        for impl in all_implementations:
            valid_impl = self._validate_and_normalize_impl(impl, resources)
            if valid_impl is not None:
                self.implementations.append(valid_impl)

        unique_impl_names = set()
        for impl in self.implementations:
            name = impl['name']
            if name in unique_impl_names:
                self._raise_benchmark_error('Duplicate implementation name: %s' % name)
            else:
                unique_impl_names.add(name)

        if self.baseline_name not in unique_impl_names:
            self._raise_benchmark_error('Benchmark config lists baseline "%s", which is not an implementation name' % self.baseline_name)

    def _run_and_validate_results(self, input_dict, impl_dict):
        # This also has the side effect of warming up the JIT so we don't benchmark compile time
        input_args = input_dict['input_args']
        input_kwargs = input_dict['input_kwargs']
        actual_results = impl_dict['function'](*input_args, **input_kwargs)
        try:
            self.validator(input_args, input_kwargs, actual_results)
        except AssertionError:
            self._raise_benchmark_error('Implementation %s failed validation on input %s' % (impl_dict['name'], input_dict['x']))

    def _timeit(self, n_iterations, input_dict, impl_dict):
        func = impl_dict['function']
        input_args = input_dict['input_args']
        input_kwargs = input_dict['input_kwargs']

        start = CLOCK_FUNCTION()
        for i in range(n_iterations):
            func(*input_args, **input_kwargs)
        stop = CLOCK_FUNCTION()

        return stop - start

    def _timeit_autoscale(self, input_dict, impl_dict, reps=3):
        # pick appropriate scale size (algo taken from Jupyter %timeit)
        for index in range(0, 10):
            number = 10 ** index
            duration = self._timeit(number, input_dict, impl_dict)
            if duration >= 0.1:
                break

        times_per_call = []
        for rep in range(reps):
            times_per_call.append(self._timeit(number, input_dict, impl_dict) / number)

        return reps, number, times_per_call

    def run_benchmark(self, quiet=False, verify_only=False):
        if not quiet:
            if verify_only:
                verb = 'Verifying'
            else:
                verb = 'Running'
            print('\n  %s %s [%s]' % (verb, self.name, self.benchmark_dir))

        x = [] # x-axis value
        # timing_resuls[category][impl_name] = dict(x, times)
        timing_results = defaultdict(lambda: defaultdict(lambda: dict(x=[], times=[])))
        for input_dict in self.input_generator():
            for impl_dict in self.implementations:
                category_str = ', '.join(input_dict['category'])
                print('    %s: %s - %s' % (impl_dict['name'], category_str, input_dict['x']), end='')
                self._run_and_validate_results(input_dict, impl_dict)

                if verify_only:
                    print(' => passed')
                else:
                    reps, iter_per_rep, times_per_call = self._timeit_autoscale (input_dict, impl_dict)


                    best_time = min(times_per_call)
                    print(' => %d reps, %d iter per rep, %f usec per call' % (reps, iter_per_rep, best_time*1e6))

                    timing_results[input_dict['category']][impl_dict['name']]['x'].append(input_dict['x'])
                    timing_results[input_dict['category']][impl_dict['name']]['times'].append(best_time)

        return timing_results

    def write_results(self, filename, bench_results):
        # Reformat bench_results for serialization to remove tuple keys to dicts
        reformatted_results = []

        for category, category_results in bench_results.items():
            tmp_category_results = dict(category_results)
            tmp_category_results['category'] = list(category)
            reformatted_results.append(tmp_category_results)

        implementations = [ dict(name=impl['name'], description=impl['description'], source=impl['source'])
            for impl in self.implementations ]

        result_dict = {
            'name': self.name,
            'description': self.description,
            'created': str(datetime.datetime.now().isoformat()),
            'xlabel': self.xlabel,
            'baseline': self.baseline_name,
            'implementations': implementations,
            'results': reformatted_results
        }
        with open(filename, 'w') as f:
            json.dump(result_dict, f)


def match_any(string, substring_list):
    for substring in substring_list:
        if substring in string:
            return True
    return False


def discover_and_run_benchmarks(source_prefix, destination_prefix, match_substrings, skip_existing=False, resources=set(),
                                verify_only=False):
    for root, dirs, files in os.walk(source_prefix):
        benchmark_subdir = os.path.relpath(root, source_prefix)
        output_dir = os.path.join(destination_prefix, benchmark_subdir)
        output_filename = os.path.join(output_dir, 'results.json')

        if Benchmark.is_benchmark_dir(root):
            if (skip_existing and os.path.exists(output_filename)) \
                     or not match_any(benchmark_subdir, match_substrings):
                print('  Skipping %s' % benchmark_subdir)
            else:
                benchmark = Benchmark(root, resources=resources)
                results = benchmark.run_benchmark(verify_only=verify_only)
                if not verify_only:
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir)
                    print('  Writing benchmark results to:', output_filename)
                    benchmark.write_results(output_filename, results)
