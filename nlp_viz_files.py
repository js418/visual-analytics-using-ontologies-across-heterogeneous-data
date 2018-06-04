'''
Visualizations for sentiment headlines.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_files.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_files

in your browser.
'''
import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,BoxSelectTool,TapTool,Range1d, Circle
from bokeh.layouts import row, column, widgetbox,layout
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,Button, RadioButtonGroup, Select, Slider,DataTable, DateFormatter, TableColumn
import networkx as nx
from datetime import datetime
import itertools
import radviz_centroid_optimization as rv
import json

def create_graph(nodes, edges, s_color,senti):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
    print(pos)
    colors = []
    labels = []
    s = []
    for v in g.nodes():
        e = str(v)
        colors.append(s_color[e])
        labels.append(e)
        s.append(senti[e])
    vx, vy = zip(*[pos[v] for v in g.nodes()])
    xs, ys = [], []
    for (a, b) in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xs.append([x0, x1])
        ys.append([y0, y1])
    return {"s1": dict(xs=xs, ys=ys), "s2": dict(vx=vx, vy=vy, labels=labels, color=colors,s=s)}

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
    s_per_df = pd.read_csv("data/entity_sentiment_avgValues/PersonSentiment.csv")
    s_pla_df = pd.read_csv("data/entity_sentiment_avgValues/PlaceSentiment.csv")
    s_org_df = pd.read_csv("data/entity_sentiment_avgValues/OrganizationSentiment.csv")
    r = 300
    margin = 80
    sentiment = ["all", "Negative", "Positive", "Neutral"]
    with open("Radviz_files.json") as json_file:
        sources = json.load(json_file)
    json_file.close()
    for s in sentiment:
        dstr = sources[s]["views"]["dstr"]
        dates = []
        for d in dstr:
            dates.append(datetime.strptime(d, "%m/%d/%Y"))
        sources[s]["views"]["s_x"] = dates

    ############# Radviz for headlines #############
    print("ploting Radviz ...")
    a_source = ColumnDataSource(data=sources["all"]["anchors"])
    v_source = ColumnDataSource(data=sources["all"]["views"])
    radviz = figure(plot_width=400, plot_height=450, title="Headlines (anchors: words), Centroid Measure",
               x_axis_type=None, y_axis_type=None,
               output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    radviz.circle(r, r, radius=r, line_color="black", fill_color=None)
    c1 = radviz.circle('a_x', 'a_y', size=7, color="#af8dc3", source=a_source)
    c2 = radviz.circle('v_x', 'v_y', size=7, color='c', alpha=0.8, source=v_source,legend= 'senti')
    radviz.x_range = Range1d(-margin, 2 * r + margin)
    radviz.y_range = Range1d(-margin, 2 * r + margin)
    c2.selection_glyph = Circle(fill_color="c", line_color=None)
    c2.nonselection_glyph = Circle(fill_color=None, line_color="c")

    hover1 = HoverTool(tooltips=[("words: ", "@anchors")], renderers=[c1])
    hover2 = HoverTool(tooltips=[("file: ", "@views"),
                                 ("headlines: ", "@h"),
                                 ("sentiment: ", "@senti"),
                                 ("sentiment score: ", "@score")
                                 ], renderers=[c2])

    radviz.add_tools(hover1)
    radviz.add_tools(hover2)

    radviz.legend.location = (0,380)
    radviz.legend.orientation = "horizontal"
    radviz.legend.label_text_font_size = "8pt"

    ############# Timeline for each file #############
    print("ploting scatter ...")
    scatter = figure(plot_width=750, plot_height=450, title="Timeline for Files",
                x_axis_type='datetime', x_axis_label='Time',y_axis_label='Sentiment Score',
               output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    c3 = scatter.circle('s_x', 'score', source=v_source, size=7, color='c', alpha=0.8, legend= 'senti')
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
    c3.selection_glyph = Circle(fill_color="c", line_color="#d73027")
    c3.nonselection_glyph = Circle(fill_color=None, line_color="c")


    #create a button group to select different sentiment headlines
    button_group = RadioButtonGroup(labels=sentiment, active=0)

    def button_group_update(attr, old, new):
        b = button_group.active
        s = sentiment[b]
        a_source.data = sources[s]["anchors"]
        v_source.data = sources[s]["views"]
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
    s4 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[],s=[]))
    n2 = figure(plot_width=600, plot_height=600, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Entities Involved in Selected Files",
                output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="black", source=s3, alpha=0.3)
    n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5,legend='s')
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s4)
    n2.legend.location = (0,580)
    n2.legend.orientation = "horizontal"
    n2.legend.label_text_font_size = "8pt"

    def update_network2(attr, old, new):
        inds = new['1d']['indices']

        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n2.title.text = "Entities Involved in Selected Files"

        if inds == []:
            return

        selected = []
        nodes = []
        edges = []
        r_per = []
        r_pla = []
        r_org = []
        t_t_edges = []
        senti = {}
        s_color = {}
        for i in inds:
            f = s2.data["labels"][i]
            selected.append(f)
            v = df.loc[df["filename"] == f]["all_nodes"].values
            n = ast.literal_eval(v[0])
            nodes = list(set(nodes).union(n))
            e = ast.literal_eval(df.loc[df["filename"] == f]["all_edges"].values[0])
            edges = list(set(edges).union(e))
            t_t_e = ast.literal_eval(df.loc[df["filename"] == f]["file_neighbors_edge"].values[0])
            t_t_edges = list(set(t_t_edges).union(t_t_e))
            r_person = df.loc[df["filename"] == f]["Person"].values[0]
            if r_person == "None":
                r_person = []
            else:
                r_person = ast.literal_eval(r_person)
            r_place = df.loc[df["filename"] == f]["Place"].values[0]
            if r_place == "None":
                r_place = []
            else:
                r_place = ast.literal_eval(r_place)
            r_organization = df.loc[df["filename"] == f]["Org"].values[0]
            if r_organization == "None":
                r_organization = []
            else:
                r_organization = ast.literal_eval(r_organization)
            r_per = list(set(r_per).union(r_person))
            r_pla = list(set(r_pla).union(r_place))
            r_org = list(set(r_org).union(r_organization))

        t_t = list(itertools.combinations(selected, 2))
        for t in t_t:
            if (t in t_t_edges) or ((t[1], t[0]) in t_t_edges):
                edges.append(t)

        for node in nodes:
            if node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment"].values[0]
            elif node in selected:
                s_color[node] = df.loc[df["filename"] == node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"] == node]["sentiment"].values[0]
            elif "-Person" in node:
                s_color[node] = "#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#f49141"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    c4.data_source.on_change("selected", update_network2)

    def update_network3(attr, old, new):
        inds = new['1d']['indices']
        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n2.title.text = "Entities Involved in Selected Files"

        if inds == []:
            return

        selected = []
        nodes = []
        edges = []
        r_per = []
        r_pla = []
        r_org = []
        t_t_edges = []
        senti = {}
        s_color = {}
        for i in inds:
            f = v_source.data["views"][i]
            selected.append(f)
            v = df.loc[df["filename"] == f]["all_nodes"].values
            n = ast.literal_eval(v[0])
            nodes = list(set(nodes).union(n))
            e = ast.literal_eval(df.loc[df["filename"] == f]["all_edges"].values[0])
            edges = list(set(edges).union(e))
            t_t_e = ast.literal_eval(df.loc[df["filename"] == f]["file_neighbors_edge"].values[0])
            t_t_edges = list(set(t_t_edges).union(t_t_e))
            r_person = df.loc[df["filename"] == f]["Person"].values[0]
            if r_person == "None":
                r_person = []
            else:
                r_person = ast.literal_eval(r_person)
            r_place = df.loc[df["filename"] == f]["Place"].values[0]
            if r_place == "None":
                r_place = []
            else:
                r_place = ast.literal_eval(r_place)
            r_organization = df.loc[df["filename"] == f]["Org"].values[0]
            if r_organization == "None":
                r_organization = []
            else:
                r_organization = ast.literal_eval(r_organization)
            r_per = list(set(r_per).union(r_person))
            r_pla = list(set(r_pla).union(r_place))
            r_org = list(set(r_org).union(r_organization))

        t_t = list(itertools.combinations(selected, 2))
        for t in t_t:
            if (t in t_t_edges) or ((t[1], t[0]) in t_t_edges):
                edges.append(t)

        for node in nodes:
            if node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment"].values[0]
            elif node in selected:
                s_color[node] = df.loc[df["filename"] == node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"] == node]["sentiment"].values[0]
            elif "-Person" in node:
                s_color[node] = "#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#f49141"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    c2.data_source.on_change("selected", update_network3)

    print("Done")

    l = column(button_group,row(radviz,scatter),row(n1,n2))

    show(l)

    curdoc().add_root(l)

main()