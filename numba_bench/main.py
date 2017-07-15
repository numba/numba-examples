#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import argparse

from .benchmark import discover_and_run_benchmarks
from .plotting import discover_and_make_plots

def main(argv):
    parser = argparse.ArgumentParser(description='Run comparative benchmarks')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output directory for benchmark results')
    parser.add_argument('-s', '--skip-existing', action='store_true', default=False,
                        help='Skip benchmarks if results already exist')
    parser.add_argument('--root', type=str, default='.',
                        help='root of benchmark directories to scan')
    parser.add_argument('benchmark_patterns', metavar='BENCHMARK', type=str,
                        nargs='*', help='match directories containing these substrings when running benchmarks.  Does not affect plotting.')
    parser.add_argument('--plot-only', action='store_true', default=False,
                        help='Only generate plots. Do not run benchmarks.')
    parser.add_argument('--run-only', action='store_true', default=False,
                        help='Only run benchmarks. Do not generate plots.')
    parser.add_argument('-r', '--resources', type=str, default='',
        help='comma separated list of resources like "gpu" that might be required by benchmarks')
    parser.add_argument('--url-root', type=str, default='https://github.com/numba/numba-examples/tree/master',
        help='Base URL on Github for benchmark source')


    args = parser.parse_args(argv[1:])

    if args.run_only and args.plot_only:
        print('Error: Cannot specify --plot-only and --run-only at same time.')
        return 1
    else:
        do_benchmark = not args.plot_only
        do_plots = not args.run_only

    root = os.path.abspath(args.root)
    output = os.path.abspath(args.output)
    resources = set(args.resources.split(','))

    if do_benchmark:
        print('Scanning %s for benchmarks' % root)

        if len(args.benchmark_patterns) == 0:
            match_substrings = ['']  # match everything
        else:
            match_substrings = args.benchmark_patterns
            print('  => matching benchmark directories: ' + ', '.join(match_substrings))

        print('Writing results to %s' % os.path.abspath(args.output))

        discover_and_run_benchmarks(root, output, match_substrings, skip_existing=args.skip_existing, resources=resources)

    if do_plots:
        print('Scanning %s for results to plot' % output)
        discover_and_make_plots(output, github_urlbase=args.url_root)


    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
