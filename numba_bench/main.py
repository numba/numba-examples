#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import argparse

from .benchmark import discover_and_run_benchmarks

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
