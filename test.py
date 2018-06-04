import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot,column,row
from bokeh.models import ColumnDataSource,HoverTool, LabelSet, Label
from bokeh.plotting import figure
import pandas as pd
import radviz_centroid_optimization as rv
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

df = pd.read_csv("nlp_entities.csv")
h_d = pd.read_csv("data/headlines.csv")
h_files = np.array(h_d["filename"])

senti = []
s_score = []
s_color = []
h =[]
for f in h_files:
    s = df.loc[df["filename"] == f]["sentiment"].values
    senti.append(s[0])
    score = df.loc[df["filename"] == f]["sentiment_score"].values
    s_score.append(score[0])
    color = df.loc[df["filename"] == f]["sentiment_color"].values
    s_color.append(color[0])
    headline = df.loc[df["filename"] == f]["headlines"].values
    h.append(headline[0])


h_d.drop(labels="headline", axis=1, inplace=True)
h_d.drop(labels="filename", axis=1, inplace=True)
h_d.drop(labels="Unnamed: 0", axis=1, inplace=True)
h_d.drop(labels="Unnamed: 0.1", axis=1, inplace=True)

words = h_d.columns.values.tolist()

g = rv.radviz_optimization(h_d.values)
g.optimize()
view = g.get_view()
anchor = g.get_anchors()

r = 300
anchor = r * anchor + r
view = r * view + r
margin = 80

a_source = ColumnDataSource(data=dict(a_x= anchor[:, 0], a_y= anchor[:, 1],anchors= words))
v_source = ColumnDataSource(data=dict(v_x= view[:, 0], v_y= view[:, 1], views= h_files, h= h, c= s_color, score=s_score,senti= senti))

print("ploting Radviz ...")
f = figure(plot_width=520, plot_height=500, title="Headlines (anchors: words), Centroid Measure",
           x_axis_type=None, y_axis_type=None, x_range=[-margin, 2 * r + margin], y_range=[-margin, 2 * r + margin],
           output_backend = "webgl")
f.circle(r, r, radius=r, line_color="black", fill_color=None)
c1 = f.circle('a_x', 'a_y', size=10, color="blue", source= a_source)
c2 = f.circle('v_x', 'v_y', size=7, color='c', alpha= 0.7, source= v_source,legend='senti')

hover1 = HoverTool(tooltips=[("words: ", "@anchors")], renderers=[c1])
hover2 = HoverTool(tooltips=[("file: ", "@views"),
                             ("headlines: ", "@h")
                             ], renderers=[c2])

f.add_tools(hover1)
f.add_tools(hover2)

X = h_d.values
perplexities = [5, 20, 35, 50]
ts = []

for i, p in enumerate(perplexities):
    print("ploting t" + str(i))
    Y = manifold.TSNE(n_components=2,perplexity=p).fit_transform(X)
    t_source = ColumnDataSource(data=dict(x=Y[:, 0], y= Y[:, 1],c= s_color, f =h_files,s=senti,h=h))
    t = figure(plot_width=300, plot_height=300, title="t-SNE, perplexity = " + str(p))
    t.circle('x', 'y', size=3, color='c', alpha= 0.6,source= t_source)
    hover = HoverTool(tooltips=[("file: ", "@f"),("headlines: ", "@h")])
    t.add_tools(hover)
    ts.append(t)

print("DONE")

l = column(f,row(ts[0],ts[1],ts[2],ts[3]))
show(l)


