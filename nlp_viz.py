'''
Visualizations for sentiment headlines.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve nlp_viz.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/nlp_viz
in your browser.
'''

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
            color.append("blue")
    labels = [str(v) for v in g.nodes()]
    vx, vy = zip(*[pos[v] for v in g.nodes()])
    xs, ys = [], []
    for (a, b) in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xs.append([x0, x1])
        ys.append([y0, y1])
    return {"s1":dict(xs=xs,ys=ys),"s2":dict(vx=vx,vy=vy,labels=labels,color=color)}

def create_file_graph(nodes,edges,df):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
    color = []
    s = []
    labels =[]
    headlines = []
    for v in g.nodes():
        f = str(v)
        labels.append(f)
        h_v = df.loc[df["filename"] == f]["headlines"].values
        headlines.append(h_v[0])
        v = df.loc[df["filename"] == f]["sentiment"].values
        s.append(v[0])
        c_v = df.loc[df["filename"] == f]["sentiment_color"].values
        color.append(c_v[0])
    vx, vy = zip(*[pos[v] for v in g.nodes()])
    xs, ys = [], []
    for (a, b) in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xs.append([x0, x1])
        ys.append([y0, y1])
    return {"s1":dict(xs=xs,ys=ys),"s2":dict(vx=vx,vy=vy,labels=labels,color=color,h=headlines,senti=s)}

def main():
    df = pd.read_csv("nlp_entities.csv")

    ############# Radviz for headlines #############
    h_d = pd.read_csv("data/headlines.csv")
    h_headlines = np.array(h_d["headline"])
    h_files = np.array(h_d["filename"])

    senti = []
    s_score = []
    s_color = []
    d = []
    e_headlines = []
    datestr = []
    for f in h_files:
        s = df.loc[df["filename"] == f]["sentiment"].values
        senti.append(s[0])
        score = df.loc[df["filename"] == f]["sentiment_score"].values
        s_score.append(score[0])
        color = df.loc[df["filename"] == f]["sentiment_color"].values
        s_color.append(color[0])

        v_d = df.loc[df["filename"] == f]["date"].values
        datestr.append(v_d[0])
        d.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
        v_h = df.loc[df["filename"] == f]["headlines"].values
        e_headlines.append(v_h[0])

    #h_d.drop(labels="headline", axis=1, inplace=True)
    #h_d.drop(labels="filename", axis=1, inplace=True)
    h_d.drop(labels="Unnamed: 0", axis=1, inplace=True)
    h_d.drop(labels="Unnamed: 0.1", axis=1, inplace=True)
    h_d.insert(len(h_d.columns), "sentiment", senti)
    h_d.insert(len(h_d.columns), "sentiment_score", s_score)
    h_d.insert(len(h_d.columns), "sentiment_color", s_color)
    h_d.insert(len(h_d.columns), "e_headlines", e_headlines)
    h_d.insert(len(h_d.columns), "file_date", d)
    h_d.insert(len(h_d.columns), "datestr", datestr)

    N = len(h_d.columns)
    new_d = h_d.iloc[:,0:(N-8)]

    words = new_d.columns.values.tolist()

    print("ploting Radviz ...")
    g = rv.radviz_optimization(new_d.values)
    g.optimize()
    view = g.get_view()
    anchor = g.get_anchors()

    r = 300
    anchor = r * anchor + r
    view = r * view + r
    margin = 80

    all_a = dict(a_x=anchor[:, 0], a_y=anchor[:, 1], anchors=words)
    all_v = dict(v_x=view[:, 0], v_y=view[:, 1], views=h_files,
                  h=h_headlines, c=s_color, score=s_score, senti=senti,
                  s_x= d,dstr= datestr, e_h= e_headlines)
    a_source = ColumnDataSource(data=all_a)
    v_source = ColumnDataSource(data=all_v)

    radviz = figure(plot_width=400, plot_height=450, title="Headlines (anchors: words), Centroid Measure",
               x_axis_type=None, y_axis_type=None,
               output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    radviz.circle(r, r, radius=r, line_color="black", fill_color=None)
    c1 = radviz.circle('a_x', 'a_y', size=7, color="blue", source=a_source)
    c2 = radviz.circle('v_x', 'v_y', size=7, color='c', alpha=0.6, source=v_source,legend= 'senti')
    radviz.x_range = Range1d(-margin, 2 * r + margin)
    radviz.y_range = Range1d(-margin, 2 * r + margin)

    hover1 = HoverTool(tooltips=[("words: ", "@anchors")], renderers=[c1])
    hover2 = HoverTool(tooltips=[("file: ", "@views"),
                                 ("headlines: ", "@h"),
                                 ("sentiment: ", "@senti"),
                                 ("sentiment score: ", "@score")
                                 ], renderers=[c2])

    radviz.add_tools(hover1)
    radviz.add_tools(hover2)

    radviz.legend.location = "top_left"

    ############# Timeline for each file #############
    print("ploting scatter ...")
    scatter = figure(plot_width=750, plot_height=450, title="Timeline for Files",
                x_axis_type='datetime', x_axis_label='Time',y_axis_label='Sentiment Score',
               output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    c3 = scatter.circle('s_x', 'score', source=v_source, size=7, color='c', alpha=0.6, legend= 'senti')
    hover3 = HoverTool(
        tooltips=[
            ("file: ", "@views"),
            ("date: ", "@dstr"),
            ("headline: ", "@e_h")
        ]
    )
    scatter.add_tools(hover3)
    scatter.x_range = Range1d(datetime.strptime('01/01/2002', "%m/%d/%Y"), datetime.strptime('12/31/2004', "%m/%d/%Y"))

    scatter.legend.location = "top_left"


    #create a button group to select different sentiment headlines
    sentiment = ["all","negative", "positive"]
    sources = {}
    for x in ["negative", "positive"]:
        data = h_d[h_d["sentiment"] == x]
        n_d = data.iloc[:, 0:(N - 8)]
        n_g = rv.radviz_optimization(n_d.values)
        n_g.optimize()
        n_view = n_g.get_view()
        n_anchor = n_g.get_anchors()
        n_anchor = r * n_anchor + r
        n_view = r * n_view + r

        a = dict(a_x=n_anchor[:, 0], a_y=n_anchor[:, 1], anchors=words)
        v = dict(v_x=n_view[:, 0], v_y=n_view[:, 1], views=np.array(data["filename"]),
                 h=np.array(data["headline"]), c=np.array(data["sentiment_color"]),
                 score=np.array(data["sentiment_score"]), senti=np.array(data["sentiment"]),
                 s_x=np.array(data["file_date"]), dstr=np.array(data["datestr"]), e_h=np.array(data["e_headlines"]))
        sources[x] = (a,v)
    sources["all"] = (all_a,all_v)

    button_group = RadioButtonGroup(labels=sentiment, active=0)

    def button_group_update(attr, old, new):
        b = button_group.active
        s = sentiment[b]
        a_source.data = sources[s][0]
        v_source.data = sources[s][1]
        return

    button_group.on_change("active", button_group_update)

    # create interactive network graph
    print("ploting network 1 ...")
    s1 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s2 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[],h=[], senti=[]))
    n1 = figure(plot_width=600, plot_height=600, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_select,box_zoom,reset,save",
                title="Neighbors of Selected Files",output_backend="webgl")
    n1.multi_line('xs', 'ys', line_color="black", source=s1, alpha=0.3)
    c4 = n1.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s2, alpha=0.5,legend='senti')
    n1.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s2)

    hover4 = HoverTool(
        tooltips=[
            ("file: ", "@labels"),
            ("headline: ", "@h")
        ], renderers= [c4]
    )
    n1.add_tools(hover4)
    n1.legend.location = "top_left"

    def update_network1(attr,old,new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        selected = []
        if inds == []:
            s1.data = dict(xs=[], ys=[])
            s2.data = dict(vx=[], vy=[], labels=[], color=[],h=[], senti=[])
            n1.title.text = "Neighbors of Selected Files"
            return
        for i in inds:
            f = v_source.data["views"][i]
            selected.append(f)
            v = df.loc[df["filename"] == f]["file_neighbors"].values
            n = ast.literal_eval(v[0])
            if f not in n:
                n.append(f)
            nodes = list(set(nodes).union(n))
            e_v = df.loc[df["filename"] == f]["file_neighbors_edge"].values
            e = ast.literal_eval(e_v[0])
            edges = list(set(edges).union(e))
        new_dict = create_file_graph(nodes,edges,df)
        s1.data = new_dict["s1"]
        s2.data = new_dict["s2"]
        n1.title.text = "Neighbors of Selected Files - " +", ".join(selected)
        return

    c2.data_source.on_change("selected",update_network1)

    print("ploting network 2 ...")
    s3 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s4 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[]))
    n2 = figure(plot_width=600, plot_height=600, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Entities Involved in Selected Files",
                output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="black", source=s3, alpha=0.3)
    n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5)
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s4)

    def update_network2(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        t_t_edges =[]
        selected = []
        if inds == []:
            s3.data = dict(xs=[], ys=[])
            s4.data = dict(vx=[], vy=[], labels=[], color=[])
            n2.title.text = "Entities Involved in Selected Files"
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
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    c4.data_source.on_change("selected", update_network2)

    def update_network3(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        t_t_edges =[]
        selected = []
        if inds == []:
            s3.data = dict(xs=[], ys=[])
            s4.data = dict(vx=[], vy=[], labels=[], color=[])
            n2.title.text = "Entities Involved in Selected Files"
            return
        for i in inds:
            f = v_source.data["views"][i]
            selected.append(f)
            v = df.loc[df["filename"] == f]["all_nodes"].values
            n = ast.literal_eval(v[0])
            nodes = list(set(nodes).union(n))
            e_v = df.loc[df["filename"] == f]["all_edges"].values
            e = ast.literal_eval(e_v[0])
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
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    c2.data_source.on_change("selected", update_network3)

    print("Done")


    """
    row1 = layout(
        children=[
            [radviz,scatter],
        ],
        sizing_mode="stretch_both")
    """

    l = column(button_group,row(radviz,scatter),row(n1,n2))

    show(l)

    curdoc().add_root(l)


main()