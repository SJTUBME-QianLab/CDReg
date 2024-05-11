import os
import pickle
import scipy
import pandas as pd
import numpy as np

import pyecharts
import pyecharts.options as opts
from pyecharts.charts import Sankey


data = pd.read_excel('./Results_LUAD.xlsx', sheet_name='Top50-vsLcon0')
data = data.iloc[:, :7]


nodes = [{'name': kk} for kk in data['IlmnID']] + list(sorted(set(data['Gene']))) + ['w C', 'w/o C']
links = []
for i in range(len(data)):
    probe, gene = data.loc[i, ['IlmnID', 'Gene']]
    links.append({'source': gene, 'target': probe, 'value': 1})
    if not np.isnan(data.loc[i, 'w/o C']):
        links.append({'source': probe, 'target': 'w/o C', 'value': 1})
    if not np.isnan(data.loc[i, 'w C']):
        links.append({'source': probe, 'target': 'w C', 'value': 1})


nodes = [{'name': kk} for kk in sorted(set(data['Gene']))]
links = []
for gg in nodes:
    gene = gg['name']
    dfi = data[data['Gene']==gene]
    Lcon = len(dfi) - sum(np.isnan(dfi['w/o C']))
    ours = len(dfi) - sum(np.isnan(dfi['w C']))
    print(gene)
    if Lcon > 0 and ours == 0:
        links.append({'source': 'w/o C', 'target': 'w/o C only', 'value': Lcon})
        links.append({'source': 'w/o C only', 'target': gene, 'value': Lcon})
    elif ours > 0 and Lcon == 0:
        links.append({'source': 'w C', 'target': 'w C only', 'value': ours})
        links.append({'source': 'w C only', 'target': gene, 'value': ours})
    elif ours > 0 and Lcon > 0:
        links.append({'source': 'w/o C', 'target': 'both', 'value': Lcon})
        links.append({'source': 'w C', 'target': 'both', 'value': ours})
        links.append({'source': 'both', 'target': gene, 'value': Lcon})
        links.append({'source': 'both', 'target': gene, 'value': ours})
nodes = [{'name': kk} for kk in ['w/o C', 'w C', 'w/o C only', 'w C only', 'both']] + nodes


c = (
    Sankey(
        init_opts=opts.InitOpts(theme=pyecharts.globals.ThemeType.WESTEROS, 
                                width='300px', height='540px', 
                                renderer='svg', bg_color=None, 
                                is_horizontal_center=True,
                               animation_opts=opts.AnimationOpts(animation=False)),
    )
    .add(
        "sankey",
        nodes,
        links,
        itemstyle_opts=opts.ItemStyleOpts(),
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="right", font_family='Arial', font_size=12, color='black'),
        pos_left = "1%",
        pos_right = "25%",
        pos_top = "1%",
        pos_bottom = "1%",
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title=""), 
        legend_opts=opts.LegendOpts(is_show=False),
    )
)
c.render_notebook()


from snapshot_pyppeteer import snapshot
import nest_asyncio
nest_asyncio.apply()
pyecharts.render.make_snapshot(snapshot, c.render(), "./LUAD/Fig4b_sankey.svg", delay=5, pixel_ratio=1)


