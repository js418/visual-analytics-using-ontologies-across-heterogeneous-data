import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,BoxSelectTool,TapTool,Range1d
from bokeh.layouts import row, column, widgetbox,layout, Spacer
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,Button, RadioButtonGroup, Select, Slider
import networkx as nx
from datetime import datetime
import itertools
import radviz_centroid_optimization as rv
from bokeh.models.tickers import FixedTicker

def main():
    df = pd.read_csv("C:\\Users\\tensa\\Desktop\\courses\\summer\\iris-species\\Iris.csv")
    species = np.array(df["Species"])
    color = []
    for s in species:
        if s=="Iris-setosa":
            color.append("pink")
        elif s == "Iris-versicolor":
            color.append("blue")
        elif s == "Iris-virginica":
            color.append("green")
    df.insert(len(df.columns), "color", color)
    new_df = df.iloc[:, 1:5]
    label = new_df.columns.values.tolist()
    x = [1,2,3,4]
    #print(df.iloc[0].values.tolist())
    n = len(df.index)
    xs=[]
    ys=[]
    for i in range(n):
        xs.append(x)
        y = new_df.iloc[i].values.tolist()
        ys.append(y)

    source = ColumnDataSource(data=dict(x=xs,y=ys, c=color, s=species))
    p = figure(plot_width=800, plot_height=400, x_range=label,y_range = [-1,9],title="Parallel Coordinates for Iris Dataset")
    p.multi_line('x','y', color='c', source= source,alpha=0.5, line_width=2)
    p.multi_line([[1,1],[2,2],[3,3],[4,4]],[[-1,9],[-1,9],[-1,9],[-1,9]],color="black")
    #p.x_range=Range1d(0,3)


    show(p)





main()