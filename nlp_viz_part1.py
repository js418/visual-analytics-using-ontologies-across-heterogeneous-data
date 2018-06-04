''' Present scatter plots with linked network graphs to show the entities involved in each file.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_part1.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_part1

in your browser.

'''

import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,BoxSelectTool,TapTool
from bokeh.layouts import row, column, widgetbox
from bokeh.io import show
from bokeh.plotting import figure, curdoc
import networkx as nx
import itertools

def create_array(df):
    l = df.tolist()
    list = []
    for i in l:
        if i == "None":
            list.append([])
        else:
            list.append(ast.literal_eval(i))
    return np.array(list)

def create_graph(nodes,edges,selected):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
    color = []
    for v in g.nodes():
        if str(v) in selected:
            color.append("red")
        elif "-Person" in str(v):
            color.append("#f441e8")
        elif "-Place" in str(v):
            color.append("#e2f441")
        elif "-Org" in str(v):
            color.append("#f49141")
        else:
            color.append("green")
    labels = [str(v) for v in g.nodes()]
    vx, vy = zip(*[pos[v] for v in g.nodes()])
    xs, ys = [], []
    for (a, b) in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xs.append([x0, x1])
        ys.append([y0, y1])
    return {"s1":dict(xs=xs,ys=ys),"s2":dict(vx=vx,vy=vy,labels=labels,color=color)}

def main():

    df = pd.read_csv("nlp_entities.csv")

    file = np.array(df["filename"])
    date = np.array(pd.to_datetime(df['date'],errors='coerce'),dtype=np.datetime64)
    datestr = np.array(df["date"])
    person = create_array(df["Person"])
    organization = create_array(df["Org"])
    location = create_array(df["Place"])
    p_n = np.array([len(i) for i in person])
    o_n = np.array([len(j) for j in organization])
    l_n = np.array([len(k) for k in location])
    keyword = np.array(df["keyword"])
    cluster = np.array(df["cluster_label"])
    c_color = np.array(df["color_label"])
    headline = np.array(df["headlines"])

    # create overall visualizations and interactive network graphs
    overall_source = ColumnDataSource(
        data=dict(x=date, y0=p_n, y1=o_n,y2=l_n,dstr=datestr,
                  p=person, o=organization, l=location,doc=file,
                  h = headline,k=keyword, c=cluster, color=c_color))

    TOOLS = "box_select,pan,box_zoom,reset"

    f1 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of persons involved in textfile",
                x_axis_type='datetime', x_axis_label='Time', y_axis_label='number of persons',output_backend="webgl",
                )
    c1=f1.circle('x', 'y0', source=overall_source, size=7,color="color", legend="c",alpha=0.4)
    hover1 = HoverTool(
        tooltips=[
            ("document: ", "@doc"),
            ("date", "@dstr"),
            ("persons", "@y0"),
            ("headline: ", "@h")
        ]
    )
    f1.add_tools(hover1)

    f2 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of organizations involved in textfile",
                x_range=f1.x_range, x_axis_type='datetime', x_axis_label='Time',
                y_axis_label='number of organizations',output_backend="webgl")
    f2.triangle('x', 'y1', source=overall_source, size=7, color="color",legend="c", alpha=0.4)
    hover2 = HoverTool(
        tooltips=[
            ("document: ", "@doc"),
            ("date", "@dstr"),
            ("organizations", "@y1"),
            ("headline: ", "@h")
        ]
    )
    f2.add_tools(hover2)

    f3 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of locations involved in textfile",
                x_range=f1.x_range, x_axis_type='datetime', x_axis_label='Time',
                y_axis_label='number of locations',output_backend="webgl")
    f3.square('x', 'y2', source=overall_source, size=7, color="color", legend="c", alpha=0.4)
    hover3 = HoverTool(
        tooltips=[
            ("document: ", "@doc"),
            ("date", "@dstr"),
            ("organizations", "@y2"),
            ("headline: ", "@h")
        ]
    )
    f3.add_tools(hover3)

    # create interactive network graph

    s1 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s2 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[]))
    n1 = figure(plot_width=600, plot_height=600, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_select,box_zoom,reset,save", title="Neighbors of Selected Files",
                output_backend="webgl")
    n1.multi_line('xs', 'ys', line_color="blue", source=s1,alpha=0.3)
    c2 = n1.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s2,alpha=0.5)
    n1.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s2)

    def update_network1(attr,old,new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        selected = []
        if inds == []:
            s1.data = dict(xs=[], ys=[])
            s2.data = dict(vx=[], vy=[], labels=[], color=[])
            return
        for i in inds:
            n = ast.literal_eval(df.loc[i]["file_neighbors"])
            f = df.loc[i]["filename"]
            selected.append(f)
            if f not in n:
                n.append(f)
            nodes = list(set(nodes).union(n))
            e = ast.literal_eval(df.loc[i]["file_neighbors_edge"])
            edges = list(set(edges).union(e))
        new_dict = create_graph(nodes,edges,selected)
        s1.data = new_dict["s1"]
        s2.data = new_dict["s2"]
        return

    c1.data_source.on_change("selected",update_network1)

    s3 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s4 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[]))
    n2 = figure(plot_width=600, plot_height=600, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Relationships of Selected Files",
                output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="blue", source=s3,alpha=0.3)
    n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5)
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle",source=s4)

    def update_network2(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        t_t_edges =[]
        selected = []
        if inds == []:
            s3.data = dict(xs=[], ys=[])
            s4.data = dict(vx=[], vy=[], labels=[], color=[])
            return
        for i in inds:
            f = s2.data["labels"][i]
            v = df.loc[df["filename"] == f]["all_nodes"].values
            n = ast.literal_eval(v[0])
            selected.append(f)
            nodes = list(set(nodes).union(n))
            e = ast.literal_eval(df.loc[df["filename"] == f]["all_edges"].values[0])
            edges = list(set(edges).union(e))
            t_t_e = ast.literal_eval(df.loc[df["filename"] == f]["file_neighbors_edge"].values[0])
            t_t_edges = list(set(t_t_edges).union(t_t_e))
        t_t = list(itertools.combinations(selected, 2))
        for t in t_t:
            if (t in t_t_edges) or ((t[1],t[0])in t_t_edges):
                edges.append(t)
        new_dict = create_graph(nodes, edges, selected)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        return

    c2.data_source.on_change("selected", update_network2)

    def update_network3(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        t_t_edges =[]
        selected = []
        if inds == []:
            s3.data = dict(xs=[], ys=[])
            s4.data = dict(vx=[], vy=[], labels=[], color=[])
            return
        for i in inds:
            n = ast.literal_eval(df.loc[i]["all_nodes"])
            f = df.loc[i]["filename"]
            selected.append(f)
            nodes = list(set(nodes).union(n))
            e = ast.literal_eval(df.loc[i]["all_edges"])
            edges = list(set(edges).union(e))
            t_t_e = ast.literal_eval(df.loc[i]["file_neighbors_edge"])
            t_t_edges = list(set(t_t_edges).union(t_t_e))
        t_t = list(itertools.combinations(selected, 2))
        for t in t_t:
            if (t in t_t_edges) or ((t[1],t[0])in t_t_edges):
                edges.append(t)
        new_dict = create_graph(nodes, edges, selected)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        return

    c1.data_source.on_change("selected", update_network3)

    layout = column(row(f1,f2,f3),row(n1,n2))
    curdoc().add_root(layout)

main()