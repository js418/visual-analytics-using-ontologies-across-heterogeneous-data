import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,OpenURL, TapTool,Range1d,Circle,Label
from bokeh.layouts import row, column, widgetbox,layout
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,RadioButtonGroup, Select, DataTable, TableColumn
import networkx as nx
from datetime import datetime
import itertools
from os import path
from wordcloud import WordCloud

d = path.dirname(__file__)
width = 600
height = 400

# Read the whole text.
text = open(path.join(d, 'data/nlp/1101163636667.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud(background_color="white",width=width, height=height).generate(text)
wc_layout = wordcloud.layout_
#print(wordcloud.layout_)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


words = [x[0][0] for x in wc_layout]
w_size = [str(x[1])+"px" for x in wc_layout]
w_color = [x[4] for x in wc_layout]
w_a = []
w_x = []
w_y = []
for x in wc_layout:
    v_x = int(x[2][1] * wordcloud.scale)
    v_y = int(x[2][0] * wordcloud.scale)
    #v_y = height - v_y - x[1]
    if x[3] == None:
        w_a.append(0)
        #v_y = height - v_y - x[1]
    else:
        w_a.append(np.pi / 2)
        #v_x = v_x + x[1]
        #v_y = height - v_y- x[1]
    w_x.append(v_x)
    w_y.append(v_y)

source = ColumnDataSource(data=dict(x= w_x, y= w_y, w = words, s = w_size, c= w_color, a= w_a))
f = figure(plot_width= 700, plot_height=500,title='Word Cloud',x_axis_type=None, y_axis_type=None)
t = f.text(x='x', y='y',text='w',text_font_size='s',text_color='c', text_font= "Arial", angle='a', source = source)

hover = HoverTool(
        tooltips=[
            ("x: ", "$x"),
            ("y: ", "$y"),
        ], renderers=[t]
    )

f.add_tools(hover)

show(f)

