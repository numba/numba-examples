import os
import unittest
import tempfile
import shutil

from bench_runner import Benchmark
import bench_runner


class TestBenchmark(unittest.TestCase):
	EXAMPLE_BENCHMARK = os.path.join('waveforms', 'zero_suppression')

	def test_is_benchmark_dir(self):
		self.assertFalse(Benchmark.is_benchmark_dir('waveforms'))
		self.assertTrue(Benchmark.is_benchmark_dir(self.EXAMPLE_BENCHMARK))

	def test_init_validate_and_normalize(self):
		# No exceptions raised
		benchmark = Benchmark(self.EXAMPLE_BENCHMARK)

	def test_run_benchmark(self):
		# No exceptions raised
		benchmark = Benchmark(self.EXAMPLE_BENCHMARK)
		results = benchmark.run_benchmark()


class TestDiscovery(unittest.TestCase):
	def setUp(self):
		self.source_dir = os.path.dirname(__file__)
		self.tmpdir = tempfile.mkdtemp(prefix='numba_examples_test_')

	def tearDown(self):
		shutil.rmtree(self.tmpdir)

	def test_discovery(self):
		bench_runner.discover_and_run_benchmarks(self.source_dir, self.tmpdir, [''])


class TestPlotting(unittest.TestCase):
	def setUp(self):
		self.source_dir = os.path.dirname(__file__)
		self.tmpdir = tempfile.mkdtemp(prefix='numba_examples_test_')
		results_dir = os.path.join(self.tmpdir, 'results', 'waveforms', 'zero_suppression')
		self.results_file = os.path.join(results_dir, 'results.json')
		os.makedirs(results_dir)
		shutil.copyfile(os.path.join(self.source_dir, 'test_results.json'), self.results_file)

	def tearDown(self):
		shutil.rmtree(self.tmpdir)

	def test_plot(self):
		bench_runner.generate_plots(self.results_file, 'test.html')

