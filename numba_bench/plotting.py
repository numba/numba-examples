#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import json
from collections import defaultdict

from bokeh import plotting
from bokeh import layouts
from bokeh.models import HoverTool, PrintfTickFormatter, Panel, Tabs, Legend
from bokeh.models.widgets import Div
from bokeh.embed import file_html
from bokeh.resources import CDN

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from jinja2 import Template

if sys.version_info <= (3, 0):
    range = xrange


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
    with open(os.path.join(os.path.dirname(__file__), 'plot.tmpl.html')) as f:
        html_template = Template(f.read())

    with open(results_filename, 'r') as f:
        results = json.load(f)

    name = results['name']
    xlabel = results['xlabel']
    baseline = results['baseline']

    # Make plot
    #plotting.output_file(plot_filename)

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

    html = file_html(layouts.column(sections), 
        resources=CDN,
        title='Example: %s' % results['name'],
        template=html_template)

    with open(plot_filename, 'w') as f:
        f.write(html)

    return results


def discover_and_make_plots(destination_prefix):
    with open(os.path.join(os.path.dirname(__file__), 'index.tmpl.html')) as f:
        index_template = Template(f.read())

    benchmark_pages = defaultdict(list)

    index_page = os.path.join(destination_prefix, 'index.html')

    for root, dirs, files in os.walk(destination_prefix):
        output_subdir = os.path.relpath(root, destination_prefix)
        results_filename = os.path.join(root, 'results.json')
        plot_filename = os.path.join(root, 'results.html')

        if os.path.exists(results_filename):
            print('  Found: %s' % results_filename)
            results = generate_plots(results_filename, plot_filename)
            benchmark_pages[os.path.dirname(output_subdir)].append(dict(name=results['name'], path=os.path.join(output_subdir, 'results.html')))

    # Generate index page
    with open(index_page, 'w') as f:
        f.write(index_template.render(sections=benchmark_pages))
