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

from bokeh import plotting
from bokeh import layouts
from bokeh.models import HoverTool, PrintfTickFormatter, Panel, Tabs, Legend
from bokeh.models.widgets import Div

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


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

    def __init__(self, benchmark_dir):
        self.benchmark_dir = benchmark_dir
        self.benchmark_config_filename = os.path.join(benchmark_dir, BENCH_CONFIG_FILENAME)
        self.python_file_cache = {}

        with open(self.benchmark_config_filename, 'r') as f:
            config = yaml.load(f)

        self._validate_and_normalize_config(config)

    def _raise_benchmark_error(self, message):
        raise BenchmarkError(self.benchmark_dir, message)

    def _load_function(self, descriptor):
        filename, function_name = descriptor.split(':')
        path = os.path.join(self.benchmark_dir, filename)

        if path not in self.python_file_cache:
            # Load Python file
            global_dict = {}
            execfile(path, global_dict)
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
            print('No match')
            return contents
        else:
            return match.group(1)

    def _validate_and_normalize_impl(self, impl):
        try:
            name = impl['name']
        except KeyError:
            self._raise_benchmark_error('Benchmark implementation missing "name" attribute')

        # optional description
        description = impl.get('description')

        try:
            function_descriptor = impl['function']
        except KeyError:
            self._raise_benchmark_error('Benchmark implementation %s missing "function" attribute' % name)

        function = self._load_function(function_descriptor)

        source = self._load_code_fragment(function_descriptor, name)

        # optional list of requirement flags
        requires = impl.get('requires', [])

        return dict(name=name, description=description, function=function, source=source, requires=requires)

    def _validate_and_normalize_config(self, config):
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
            implementations = config['implementations']
        except KeyError:
            self._raise_benchmark_error('Benchmark config missing "implementations" attribute')

        self.implementations = [
            self._validate_and_normalize_impl(impl)
            for impl in implementations
        ]

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
            if duration >= 0.2:
                break

        times_per_call = []
        for rep in range(reps):
            times_per_call.append(self._timeit(number, input_dict, impl_dict) / number)

        return reps, number, times_per_call

    def run_benchmark(self, quiet=False):
        if not quiet:
            print('\n  Running %s [%s]' % (self.name, self.benchmark_dir))

        x = [] # x-axis value
        # timing_resuls[category][impl_name] = dict(x, times)
        timing_results = defaultdict(lambda: defaultdict(lambda: dict(x=[], times=[])))
        for input_dict in self.input_generator():
            for impl_dict in self.implementations:
                category_str = ', '.join(input_dict['category'])
                print('    %s: %s - %s' % (impl_dict['name'], category_str, input_dict['x']), end='')
                self._run_and_validate_results(input_dict, impl_dict)
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


def discover_and_run_benchmarks(source_prefix, destination_prefix, match_substrings, skip_existing=False):
    for root, dirs, files in os.walk(source_prefix):
        benchmark_subdir = os.path.relpath(root, source_prefix)
        output_dir = os.path.join(destination_prefix, benchmark_subdir)
        output_filename = os.path.join(output_dir, 'results.json')

        if Benchmark.is_benchmark_dir(root):
            if (skip_existing and os.path.exists(output_filename)) \
                     or not match_any(benchmark_subdir, match_substrings):
                print('  Skipping %s' % benchmark_subdir)
            else:
                benchmark = Benchmark(root)
                results = benchmark.run_benchmark()
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                print('  Writing benchmark results to:', output_filename)
                benchmark.write_results(output_filename, results)


IMPL_STYLES = ['solid', 'dashed', 'dotted']
IMPL_COLORS = ['blue', 'green', 'red', 'orange']
WIDTH = 800


def make_plot(results, title, xlabel, ylabel, baseline, ycolname, yaxis_format):
    p = plotting.figure(plot_width=WIDTH, plot_height=250, title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        toolbar_location="above", tools='box_zoom,reset')

    legend_items = []

    baseline_times = [t * 1e6 for t in results[baseline]['times']]

    for i, (impl_name, impl_data) in enumerate(results.items()):
        color = IMPL_COLORS[i % len(IMPL_COLORS)]
        style = IMPL_STYLES[i % len(IMPL_STYLES)]

        data = dict(x=impl_data['x'])
        # convert to microseconds
        data['times'] = [t * 1e6 for t in impl_data['times']]
        data['name'] = [impl_name] * len(data['x'])
        data['speedup'] = [b/t for (t,b) in zip(data['times'], baseline_times)]
        # not this is items/sec
        data['throughput'] = [items/t for (t, items) in zip(impl_data['times'], impl_data['x'])]

        source = plotting.ColumnDataSource(data=data)
        line = p.line('x', ycolname, source=source,
            line_width=2, line_color=color, line_dash=style)
        marker = p.circle('x', ycolname, source=source,
            size=10, fill_color=color)
        legend_items.append( (impl_name, [line, marker]) )

    hover = HoverTool(
        tooltips=[
            ('implementation', '@name'),
            ('x', '@x{%1.0e}'),
            ('time per call', '@times{%1.1e} usec'),
            ('speedup', '@{speedup}{%1.1f}x'),
            ('throughput', '@{throughput}{%1.1e}'),
        ],
        formatters={
            'x': 'printf',
            'times': 'printf',
            'speedup': 'printf',
            'throughput': 'printf',
        }
    )
    p.add_tools(hover)
    p.xaxis[0].formatter = PrintfTickFormatter(format='%1.0e')
    p.yaxis[0].formatter = PrintfTickFormatter(format=yaxis_format)

    legend = Legend(items=legend_items, location=(0, -30))
    p.add_layout(legend, 'right')

    return p


def generate_plots(results_filename, plot_filename):
    with open(results_filename, 'r') as f:
        results = json.load(f)

    name = results['name']
    xlabel = results['xlabel']
    baseline = results['baseline']

    # Make plot
    plotting.output_file(plot_filename)

    sections = [
        layouts.row(Div(text='''<h1>Example: %(name)s</h1>
            <p><b>Description</b>: %(description)s</p>''' % results, width=WIDTH)),
    ]
    # Implementations
    sections.append(layouts.row(Div(text='<h2>Implementations</h2>', width=WIDTH)))
    source_tabs = []
    for impl in results['implementations']:
        # FIXME: Once we switch to Jinja2, put CSS classes back
        highlighted = highlight(impl['source'], PythonLexer(), HtmlFormatter(noclasses=True))
        source_tabs.append((impl['name'], Div(text=highlighted, width=WIDTH)))

    tabs = Tabs(tabs=[Panel(child=st[1], title=st[0]) for st in source_tabs], width=WIDTH)
    sections.append(layouts.row(tabs))

    # Benchmarks
    sections.append(layouts.row(Div(text='<h2>Benchmarks</h2>')))
    for category_results in results['results']:
        category = category_results['category']
        del category_results['category']

        plot_title = name + ': ' + ', '.join(category)

        speedup_p = make_plot(category_results, title=plot_title,
            xlabel=xlabel, ylabel='Speedup over %s' % baseline,
            baseline=baseline, ycolname='speedup',
            yaxis_format='%1.1f')
        throughput_p = make_plot(category_results, title=plot_title,
            xlabel=xlabel, ylabel='%s / sec' % xlabel,
            baseline=baseline, ycolname='throughput',
            yaxis_format='%1.0e')
        raw_p = make_plot(category_results, title=plot_title,
            xlabel=xlabel, ylabel='Execution time (usec)',
            baseline=baseline, ycolname='times',
            yaxis_format='%1.0f')
        
        tabs = Tabs(tabs=[
            Panel(child=speedup_p, title='Speedup'), 
            Panel(child=throughput_p, title='Throughput'), 
            Panel(child=raw_p, title='Raw times')
        ], width=WIDTH)
        sections.append(layouts.row(tabs))

    plotting.save(layouts.column(sections))


def main(argv):
    parser = argparse.ArgumentParser(description='Run comparative benchmarks')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output directory for benchmark results')
    parser.add_argument('-s', '--skip-existing', action='store_true', default=False,
                        help='Skip benchmarks if results already exist')
    parser.add_argument('--root', type=str, default='.',
                        help='root of benchmark directories to scan')
    parser.add_argument('benchmark_patterns', metavar='BENCHMARK', type=str,
                        nargs='*', help='match benchmark directories containing these substrings')

    args = parser.parse_args(argv[1:])

    root = os.path.abspath(args.root)
    print('Scanning %s for benchmarks' % root)

    if len(args.benchmark_patterns) == 0:
        match_substrings = ['']  # match everything
    else:
        match_substrings = args.benchmark_patterns
        print('  => matching benchmark directories: ' + ', '.join(match_substrings))

    output = os.path.abspath(args.output)
    print('Writing results to %s' % os.path.abspath(args.output))

    discover_and_run_benchmarks(root, output, match_substrings, skip_existing=args.skip_existing)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
