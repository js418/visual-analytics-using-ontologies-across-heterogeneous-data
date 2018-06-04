'''
Search each entity by index or name.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_part2.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_part2

in your browser.
'''

import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,BoxSelectTool,TapTool,Range1d
from bokeh.layouts import row, column, widgetbox,layout
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,Button, RadioButtonGroup, Select, Slider
import networkx as nx
from datetime import datetime
import itertools

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
            color.append("#a1d76a")
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
    per_df = pd.read_csv("Person.csv")
    pla_df = pd.read_csv("Place.csv")
    org_df = pd.read_csv("Org.csv")

    entities = ["Person", "Place", "Org"]
    relations = [["Person_Org","Person_Place","Person_Person"],["Place_Person","Place_Org","Place_Place"],["Org_Person","Org_Place","Org_Org"]]
    dfs = [per_df, pla_df,org_df]

    button_group = RadioButtonGroup(labels=entities, active=0)
    slider = Slider(title="by Index", value=0, start=0, end=len(per_df.index)-1, step=1)
    text = TextInput(title="by Name:", value='')
    select1 = Select(title="y axis:", value="Person", options=["Person", "Place", "Org"])

    files = per_df.iloc[0]["Documents"]
    d = []
    y_n = []
    headlines = []
    datestr = []
    if files == "None":
        files = []
    else:
        files = ast.literal_eval(files)
        for f in files:
            v_d = df.loc[df["filename"] == f]["date"].values
            datestr.append(v_d[0])
            d.append(datetime.strptime(v_d[0],"%m/%d/%Y"))
            v_p = df.loc[df["filename"] == f]["Person"].values
            y_n.append(len(ast.literal_eval(v_p[0])))
            v_h = df.loc[df["filename"] == f]["headlines"].values
            headlines.append(v_h[0])
    dates = np.array(d)
    n = np.array(y_n)
    headlines = np.array(headlines)
    name = per_df.iloc[0]["Person"]

    # create the timeline plot for the searched entity
    s_source = ColumnDataSource(data=dict(x = dates, y = n, f = files, h =headlines, d = datestr))
    TOOLS = "box_select,pan,box_zoom,reset"
    f1 = figure(tools=TOOLS, plot_width=600, plot_height=450, title="Timeline for '" + name + "'",
                x_axis_type='datetime', x_axis_label='Time', y_range=[-50,250],
                y_axis_label='number in total', output_backend="webgl")
    c1 = f1.circle('x', 'y', source=s_source, size=10, color="red", alpha=0.5)
    hover1 = HoverTool(
        tooltips=[
            ("document: ", "@f"),
            ("date", "@d"),
            ("headline: ", "@h")
        ]
    )
    f1.add_tools(hover1)
    f1.x_range = Range1d(datetime.strptime('01/01/2002',"%m/%d/%Y"),datetime.strptime('12/31/2004',"%m/%d/%Y"))

    def update_y(attr,old,new):
        y = select1.value
        new_y = []
        for f in s_source.data["f"]:
            v = df.loc[df["filename"] == f][y].values
            new_y.append(len(ast.literal_eval(v[0])))
        new_y = np.array(new_y)
        s_source.data["y"] = new_y
        return

    select1.on_change("value",update_y)

    # create interactive network graph for timeline plot
    s1 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s2 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[]))
    n1 = figure(plot_width=500, plot_height=500, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",title="Related Entities of Selected Files",
                output_backend="webgl")
    n1.multi_line('xs', 'ys', line_color="blue", source=s1, alpha=0.3)
    c2 = n1.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s2, alpha=0.5)
    n1.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s2)

    # update network graph when the files are selected
    def update_network1(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        t_t_edges = []
        selected = []
        if inds == []:
            s1.data = dict(xs=[], ys=[])
            s2.data = dict(vx=[], vy=[], labels=[], color=[])
            return
        for i in inds:
            f = s_source.data["f"][i]
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
        s1.data = new_dict["s1"]
        s2.data = new_dict["s2"]
        return

    c1.data_source.on_change("selected", update_network1)

    # create person-place network graph
    v1 = per_df.at[0, "Person_Place"]
    edges_1 = ast.literal_eval(v1)
    nodes_1 = [name]
    for e in edges_1:
        nodes_1.append(e[1])

    g1 = create_graph(nodes_1,edges_1,[name])
    s3 = ColumnDataSource(data=g1["s1"])
    s4 = ColumnDataSource(data=g1["s2"])
    n2 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Person_Place Relationships for Person: " + name,
                output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="blue", source=s3, alpha=0.3)
    c3 = n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5)
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s4)

    # create person-org network graph
    v2 = per_df.at[0, "Person_Org"]
    edges_2 = ast.literal_eval(v2)
    nodes_2 = [name]
    for e in edges_2:
        nodes_2.append(e[1])

    g2 = create_graph(nodes_2, edges_2, [name])
    s5 = ColumnDataSource(data=g2["s1"])
    s6 = ColumnDataSource(data=g2["s2"])
    n3 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Person_Org Relationships for Person: " + name,
                output_backend="webgl")
    n3.multi_line('xs', 'ys', line_color="blue", source=s5, alpha=0.3)
    c4 = n3.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s6, alpha=0.5)
    n3.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s6)

    # create person-org network graph
    v3 = per_df.at[0, "Person_Person"]
    edges_3 = ast.literal_eval(v3)
    nodes_3 = [name]
    for e in edges_3:
        nodes_3.append(e[1])

    g3 = create_graph(nodes_3, edges_3, [name])
    s7 = ColumnDataSource(data=g3["s1"])
    s8 = ColumnDataSource(data=g3["s2"])
    n4 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Person Relationships for Person: " + name,
                output_backend="webgl")
    n4.multi_line('xs', 'ys', line_color="blue", source=s7, alpha=0.3)
    c5 = n4.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s8, alpha=0.5)
    n4.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s8)


    #update visualizations when button group changes
    def button_group_update(attr, old, new):
        b = button_group.active
        entity = entities[b]
        d = dfs[b]
        t = text.value
        #change slider
        slider.end = len(d.index) - 1
        if (slider.value>slider.end):
            slider.value = slider.end

        if t == '':
            slider_update(attr, old, new)
        else:
            text_update(attr, old, new)

    button_group.on_change("active", button_group_update)

    def slider_update(attr,old,new):
        text.value = ''

        b = button_group.active
        entity = entities[b]
        d = dfs[b]
        s = slider.value

        # clear the visualizations
        sources = [[s1, s2], [s3, s4], [s5, s6], [s7, s8]]
        networks = [n1, n2, n3, n4]
        s_source.data = dict(x=[], y=[], f=[], h=[], d=[])
        for i in range(4):
            sources[i][0].data = dict(xs=[], ys=[])
            sources[i][1].data = dict(vx=[], vy=[], labels=[], color=[])

        name = d.iloc[s][entity]

        # update cooccurence relationships
        for i in range(3):
            v = d.at[s, relations[b][i]]
            edges = ast.literal_eval(v)
            nodes = [name]
            for e in edges:
                nodes.append(e[1])
            g = create_graph(nodes, edges, [name])
            sources[i + 1][0].data = g["s1"]
            sources[i + 1][1].data = g["s2"]
            networks[i + 1].title.text = relations[b][i] + " Relationships for " + entity + ": " + name

        #update timeline plot
        files = d.iloc[s]["Documents"]
        dates = []
        y_n = []
        headlines = []
        datestr = []
        select = select1.value
        if files == "None":
            files = []
            f1.title.text = "No files found for '" + name + "'"
        else:
            files = ast.literal_eval(files)
            for f in files:
                v_d = df.loc[df["filename"] == f]["date"].values
                datestr.append(v_d[0])
                dates.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
                v_p = df.loc[df["filename"] == f][select].values
                y_n.append(len(ast.literal_eval(v_p[0])))
                v_h = df.loc[df["filename"] == f]["headlines"].values
                headlines.append(v_h[0])
        dates = np.array(dates)
        n = np.array(y_n)
        headlines = np.array(headlines)
        s_source.data = dict(x=dates, y=n, f=files, h=headlines, d=datestr)
        f1.title.text = "Timeline for '" + name + "'"
        return

    slider.on_change("value", slider_update)

    def text_update(attr,old,new):
        b = button_group.active
        entity = entities[b]
        print(entity)
        d = dfs[b]
        t = text.value
        slider.value = 0

        # clear the visualizations
        sources = [[s1, s2], [s3, s4], [s5, s6], [s7, s8]]
        networks = [n1, n2, n3, n4]
        s_source.data = dict(x=[], y=[], f=[], h=[], d=[])
        for i in range(4):
            sources[i][0].data = dict(xs=[], ys=[])
            sources[i][1].data = dict(vx=[], vy=[], labels=[], color=[])

        name = t
        if t in np.array(d[entity]):
            s = d[d[entity] == t].index.tolist()[0]
        else:
            f1.title.text = "No such " + entity + " in the datasets"
            for x in networks:
                x.title.text = "No such " + entity + " in the datasets"
            return

        # update cooccurence relationships
        for i in range(3):
            v = d.at[s, relations[b][i]]
            edges = ast.literal_eval(v)
            nodes = [name]
            for e in edges:
                nodes.append(e[1])
            g = create_graph(nodes, edges, [name])
            sources[i + 1][0].data = g["s1"]
            sources[i + 1][1].data = g["s2"]
            networks[i + 1].title.text = relations[b][i] + " Relationships for " + entity + ": " + name

        # update timeline plot
        files = d.iloc[s]["Documents"]
        dates = []
        y_n = []
        headlines = []
        datestr = []
        select = select1.value
        if files == "None":
            files = []
            f1.title.text = "No files found for '" + name + "'"
        else:
            files = ast.literal_eval(files)
            for f in files:
                v_d = df.loc[df["filename"] == f]["date"].values
                datestr.append(v_d[0])
                dates.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
                v_p = df.loc[df["filename"] == f][select].values
                y_n.append(len(ast.literal_eval(v_p[0])))
                v_h = df.loc[df["filename"] == f]["headlines"].values
                headlines.append(v_h[0])
        dates = np.array(dates)
        n = np.array(y_n)
        headlines = np.array(headlines)
        s_source.data = dict(x=dates, y=n, f=files, h=headlines, d=datestr)
        f1.title.text = "Timeline for '" + name + "'"
        return

    text.on_change("value",text_update)

    widgets = widgetbox(button_group,slider,text)
    layout = column(widgets,row(n2,n3,n4),select1,row(f1,n1))

    curdoc().add_root(layout)
    show(layout)

main()