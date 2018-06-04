'''
Search each entity by index or name.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_search.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_search

in your browser.
'''

import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,OpenURL, TapTool,Range1d,Circle
from bokeh.layouts import row, column, widgetbox,layout
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,RadioButtonGroup, Select, DataTable, TableColumn
import networkx as nx
from datetime import datetime
import itertools
from os import path
from wordcloud import WordCloud

def create_array(df):
    l = df.tolist()
    list = []
    for i in l:
        if i == "None":
            list.append([])
        else:
            list.append(ast.literal_eval(i))
    return np.array(list)

def create_scatter_source(new_df):
    new_files = np.array(new_df['filename'])
    new_date = np.array(pd.to_datetime(new_df['date'], errors='coerce'), dtype=np.datetime64)
    new_datestr = np.array(new_df["date"])
    new_headline = np.array(new_df["headlines"])
    new_senti = np.array(new_df["sentiment"])
    new_s_score = np.array(new_df["sentiment_score"])
    new_s_color = np.array(new_df["sentiment_color"])
    data = dict(x=new_date, y=new_s_score, f=new_files, dstr=new_datestr, h=new_headline,
                                         s=new_senti, c=new_s_color)
    return data

def create_graph(nodes, edges, s_color,senti):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g)
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
    entity = ["Person", "Place", "Org"]
    sentiment = ["all", "Negative", "Positive", "Neutral"]
    per_df = pd.read_csv("Person.csv")
    pla_df = pd.read_csv("Place.csv")
    org_df = pd.read_csv("Org.csv")
    s_per_df = pd.read_csv("data/entity_sentiment_avgValues/PersonSentiment.csv")
    s_pla_df = pd.read_csv("data/entity_sentiment_avgValues/PlaceSentiment.csv")
    s_org_df = pd.read_csv("data/entity_sentiment_avgValues/OrganizationSentiment.csv")
    e_dfs = {"Person": per_df, "Place": pla_df, "Org": org_df}
    s_dfs = {"Person": s_per_df, "Place": s_pla_df, "Org": s_org_df}
    files = np.array(df['filename']).tolist()
    ir_df1 = pd.read_csv("data/words_file_retrieval.csv")
    ir_df2 = pd.read_csv("data/words_file_retrieval_desc.csv")
    frames = [ir_df1, ir_df2]
    ir_df = pd.concat(frames)
    keywords = list(set(np.array(ir_df['token']).tolist()))
    keywords.sort()

    button_group_f = RadioButtonGroup(labels=sentiment, active=0)
    select_f = Select(title="By Files: ", value="all", options=["","all"]+files)
    text_f = TextInput(title="By Filenames Input (separate by ','):", value="")
    select_k = Select(title="By Keywords: ", value="", options=[""]+keywords)
    text_k = TextInput(title="By Keywords Input (separate by ','):", value="")

    ############# Timeline for each file #############
    print("ploting scatter ...")
    date = np.array(pd.to_datetime(df['date'], errors='coerce'), dtype=np.datetime64)
    datestr = np.array(df["date"])
    persons = create_array(df["Person"])
    orgs = create_array(df["Org"])
    places = create_array(df["Place"])
    e_list = {"Person": persons, "Place": places, "Org": orgs}
    headline = np.array(df["headlines"])
    senti = np.array(df["sentiment"])
    s_score = np.array(df["sentiment_score"])
    s_color = np.array(df["sentiment_color"])

    s_source = ColumnDataSource(data=dict(x=date,y=s_score,f=files, dstr=datestr,h=headline,
                                          s=senti,c=s_color))
    scatter = figure(plot_width=800, plot_height=500, title="Timeline for all Files",
                     x_axis_type='datetime', x_axis_label='Time', y_axis_label='Sentiment Score',
                     output_backend="webgl", tools="pan,tap,box_zoom,box_select,reset,save")
    s_c = scatter.circle('x', 'y', source=s_source, size=7, color='c', alpha=0.8, legend='s')
    hover1 = HoverTool(
        tooltips=[
            ("file", "@f"),
            ("date", "@dstr"),
            ("headline", "@h"),
            ("sentiment", "@s"),
        ]
    )
    scatter.add_tools(hover1)
    scatter.x_range = Range1d(datetime.strptime('12/01/2001', "%m/%d/%Y"), datetime.strptime('01/31/2005', "%m/%d/%Y"))
    scatter.legend.location = "top_left"
    s_c.selection_glyph = Circle(fill_color='c', line_color="#d73027")
    s_c.nonselection_glyph = Circle(fill_color=None, line_color="c")

    url = "http://www-edlab.cs.umass.edu/~jiesong/OntologyDrivenVisualization/data/nlp/@f"
    taptool = scatter.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    t1_source = ColumnDataSource(data=dict(f=[],h=[]))
    columns = [
        TableColumn(field='f', title="Filename"),
        TableColumn(field='h', title="Headline"),
    ]
    data_table1 = DataTable(source=t1_source, columns=columns,width= 500, height = 400)

    # update data table 1
    def datatable1_update(attr, old, new):
        inds = new['1d']['indices']
        t1_source.data = dict(f=[],h=[])
        if inds == []:
            return
        selected = [s_source.data["f"][i] for i in inds]
        h = [s_source.data["h"][i] for i in inds]
        t1_source.data = dict(f=selected,h=h)
        return

    s_c.data_source.on_change("selected", datatable1_update)

    wc1_source = ColumnDataSource(data=dict(x=[], y=[], w=[], s=[], c=[], a=[]))
    wordcloud1 = figure(plot_width=600, plot_height=400, title='Word Cloud for Selected Files', x_axis_type=None, y_axis_type=None)
    wordcloud1.text(x='x', y='y', text='w', text_font_size='s', text_color='c', text_font="Arial", angle='a', source=wc1_source)

    #callbacks for world cloud when file is selected in scatter plot
    def wc1_s_update(attr, old,new):
        inds = new['1d']['indices']

        if inds == []:
            wc1_source.data = dict(x=[], y=[], w=[], s=[], c=[], a=[])
            wordcloud1.title.text = 'Word Cloud for Selected File'
        elif len(inds) == 1:
            i = inds[0]
            f = s_source.data["f"][i]
            d = path.dirname(__file__)
            text = open(path.join(d, "data/nlp/" + f)).read()
            wordcloud = WordCloud(background_color="white", width=500, height=300).generate(text)
            wc_layout = wordcloud.layout_
            words = [x[0][0] for x in wc_layout]
            w_size = [str(x[1]) + "px" for x in wc_layout]
            w_color = [x[4] for x in wc_layout]
            w_x = [int(x[2][1] * wordcloud.scale) for x in wc_layout]
            w_y = [int(x[2][0] * wordcloud.scale) for x in wc_layout]
            w_a =[]
            for x in wc_layout:
                if x[3] == None:
                    w_a.append(0)
                else:
                    w_a.append(np.pi / 2)
            wc1_source.data = dict(x= w_x, y= w_y, w = words, s = w_size, c= w_color, a= w_a)
            wordcloud1.title.text = "Word Cloud for Selected File: " + f
        return

    s_c.data_source.on_change("selected", wc1_s_update)

    # callbacks for world cloud when file is selected in scatter plot
    def wc1_t_update(attr, old, new):
        inds = new['1d']['indices']
        wc1_source.data = dict(x=[], y=[], w=[], s=[], c=[], a=[])
        wordcloud1.title.text = 'Word Cloud for Selected File'

        if len(inds) == 1:
            i = inds[0]
            f = t1_source.data["f"][i]
            d = path.dirname(__file__)
            text = open(path.join(d, "data/nlp/" + f)).read()
            wordcloud = WordCloud(background_color="white", width=500, height=300).generate(text)
            wc_layout = wordcloud.layout_
            words = [x[0][0] for x in wc_layout]
            w_size = [str(x[1]) + "px" for x in wc_layout]
            w_color = [x[4] for x in wc_layout]
            w_x = [int(x[2][1] * wordcloud.scale) for x in wc_layout]
            w_y = [int(x[2][0] * wordcloud.scale) for x in wc_layout]
            w_a = []
            for x in wc_layout:
                if x[3] == None:
                    w_a.append(0)
                else:
                    w_a.append(np.pi / 2)
            wc1_source.data = dict(x=w_x, y=w_y, w=words, s=w_size, c=w_color, a=w_a)
            wordcloud1.title.text = "Word Cloud for Selected File: " + f
        return

    data_table1.source.on_change("selected", wc1_t_update)

    # callback for button group
    def f_sentiment_update(attr,old,new):
        s_f = select_f.value
        s_k = select_k.value
        t_f = text_f.value
        t_k = text_k.value

        if s_f != "":
            select_f_update(attr,old,new)
        elif t_f != "":
            text_f_update(attr,old,new)
        elif s_k != "":
            select_k_update(attr,old,new)
        elif t_k != "":
            text_k_update(attr,old,new)
        return

    button_group_f.on_change("active", f_sentiment_update)

    #callbacks for file selection
    def select_f_update(attr,old,new):
        s_f = select_f.value
        select_k.value = ""
        text_f.value = ""
        text_k.value = ""
        b = button_group_f.active
        s = sentiment[b]

        s1.data = dict(xs=[], ys=[])
        s2.data = dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[])
        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n1.title.text = "Neighbors of Selected Files"
        n2.title.text = "Entities Involved in Selected Files"
        t1_source.data = dict(f=[], h=[])

        if s_f == "":
            s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
            scatter.title.text = "Timeline for all Files"
            scatter.yaxis.axis_label = "Sentiment Score"
        else:
            if s_f == "all":
                if s == "all":
                    s_source.data = dict(x=date, y=s_score, f=files, dstr=datestr, h=headline,
                                         s=senti, c=s_color)
                else:
                    new_df = df.loc[df["sentiment"] == s]
                    s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for all Files"
                scatter.yaxis.axis_label="Sentiment Score"
            else:
                new_df = df.loc[df["filename"] == s_f]
                s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for File: " + s_f
                scatter.yaxis.axis_label = "Sentiment Score"
        select_f.value = s_f
        return

    select_f.on_change("value", select_f_update)

    #callbacks for file input search
    def text_f_update(attr,old,new):
        t_f = text_f.value
        select_f.value = ""
        select_k.value = ""
        text_k.value = ""
        b = button_group_f.active
        s = sentiment[b]

        s1.data = dict(xs=[], ys=[])
        s2.data = dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[])
        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n1.title.text = "Neighbors of Selected Files"
        n2.title.text = "Entities Involved in Selected Files"
        t1_source.data = dict(f=[], h=[])

        if t_f == "":
            s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
            scatter.title.text = "Timeline for all Files"
            scatter.yaxis.axis_label = "Sentiment Score"
        else:
            file_list = t_f.split(",")
            file_list = [x.strip() for x in file_list]
            for f in file_list:
                if f not in files:
                    scatter.title.text = f + " is not a valid file name in the dataset"
                    s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
                    return
            new_df = df.loc[df["filename"].isin(file_list)]
            if s == "all":
                s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for Files: " + ", ".join(file_list)
                scatter.yaxis.axis_label = "Sentiment Score"
            else:
                new_df = new_df.loc[new_df["sentiment"] == s]
                file_list = np.array(new_df["filename"]).tolist()
                s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for Files: " + ", ".join(file_list)
                scatter.yaxis.axis_label = "Sentiment Score"
        text_f.value = t_f
        return

    text_f.on_change("value", text_f_update)

    # callbacks for keyword selection
    def select_k_update(attr, old, new):
        s_k = select_k.value
        select_f.value = ""
        text_f.value = ""
        text_k.value = ""
        b = button_group_f.active
        s = sentiment[b]

        s1.data = dict(xs=[], ys=[])
        s2.data = dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[])
        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n1.title.text = "Neighbors of Selected Files"
        n2.title.text = "Entities Involved in Selected Files"
        t1_source.data = dict(f=[], h=[])

        if s_k == "":
            s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
            scatter.title.text = "Timeline for all Files"
            scatter.yaxis.axis_label = "Sentiment Score"
        else:
            l = ast.literal_eval(ir_df.loc[ir_df["token"] == s_k]["sorted_files"].values[0])
            file_list = [x[0][0] for x in l if (x[0][0] != "Description.txt") & (x[0][0] != ".1101243338500.txt.swp")]
            file_score = [x[1] for x in l if (x[0][0] != "Description.txt") & (x[0][0] != ".1101243338500.txt.swp")]
            new_df = df.loc[df["filename"].isin(file_list)]
            if s == "all":
                s_source.data = create_scatter_source(new_df)
                s_source.data["y"] = file_score
                scatter.title.text = "Timeline for Files containing keyword: " + s_k
                scatter.yaxis.axis_label = "Term Frequency"
            else:
                new_df = new_df.loc[new_df["sentiment"] == s]
                new_file_list = np.array(new_df["filename"]).tolist()
                index = [file_list.index(x) for x in new_file_list]
                new_file_score = [file_score[i] for i in index]
                s_source.data = create_scatter_source(new_df)
                s_source.data["y"] = new_file_score
                scatter.title.text = "Timeline for Files containing keyword: " + s_k
                scatter.yaxis.axis_label = "Term Frequency"
        select_k.value = s_k
        return

    select_k.on_change("value", select_k_update)

    # callbacks for keyword input search
    def text_k_update(attr, old, new):
        t_k = text_k.value
        select_f.value = ""
        select_k.value = ""
        text_f.value = ""
        b = button_group_f.active
        s = sentiment[b]

        s1.data = dict(xs=[], ys=[])
        s2.data = dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[])
        s3.data = dict(xs=[], ys=[])
        s4.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n1.title.text = "Neighbors of Selected Files"
        n2.title.text = "Entities Involved in Selected Files"
        t1_source.data = dict(f=[],h=[])

        if t_k == "":
            s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
            scatter.title.text = "Timeline for all Files"
            scatter.yaxis.axis_label = "Sentiment Score"
        else:
            tokens = t_k.split(",")
            f_list = []
            for k in tokens:
                k = k.strip()
                if k not in keywords:
                    scatter.title.text = k + " is not a valid keyword in the dataset"
                    s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
                    return
                l = ast.literal_eval(ir_df.loc[ir_df["token"] == k]["sorted_files"].values[0])
                f_list.append([x[0][0] for x in l if (x[0][0] != "Description.txt") & (x[0][0] != ".1101243338500.txt.swp")])
            file_list = f_list[0]
            for f in f_list:
                file_list = list(set(file_list).intersection(f))
            if file_list == []:
                scatter.title.text = "No files are found to containing all the keywords: " + ", ".join(tokens)
                s_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
                return
            new_df = df.loc[df["filename"].isin(file_list)]
            if s == "all":
                s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for Files containing keywords: " + ", ".join(tokens)
                scatter.yaxis.axis_label = "Sentiment Score"
            else:
                new_df = new_df.loc[new_df["sentiment"] == s]
                s_source.data = create_scatter_source(new_df)
                scatter.title.text = "Timeline for Files containing keywords: " + ", ".join(tokens)
                scatter.yaxis.axis_label = "Sentiment Score"
        text_k.value = t_k
        return

    text_k.on_change("value", text_k_update)

    # create interactive network graph
    print("ploting network 1 ...")
    s1 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s2 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[]))
    n1 = figure(plot_width=500, plot_height=500, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_select,box_zoom,reset,save",
                title="Neighbors of Selected Files", output_backend="webgl")
    n1.multi_line('xs', 'ys', line_color="black", source=s1, alpha=0.3)
    n1_c = n1.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s2, alpha=0.5, legend='senti')
    n1.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s2)

    hover2 = HoverTool(
        tooltips=[
            ("file: ", "@labels"),
            ("headline: ", "@h")
        ], renderers=[n1_c]
    )
    n1.add_tools(hover2)
    n1.legend.location = "top_left"

    def update_network1(attr, old, new):
        inds = new['1d']['indices']
        nodes = []
        edges = []
        selected = []
        if inds == []:
            s1.data = dict(xs=[], ys=[])
            s2.data = dict(vx=[], vy=[], labels=[], color=[], h=[], senti=[])
            n1.title.text = "Neighbors of Selected Files"
            return
        for i in inds:
            f = s_source.data["f"][i]
            selected.append(f)
            v = df.loc[df["filename"] == f]["file_neighbors"].values
            n = ast.literal_eval(v[0])
            if f not in n:
                n.append(f)
            nodes = list(set(nodes).union(n))
            e_v = df.loc[df["filename"] == f]["file_neighbors_edge"].values
            e = ast.literal_eval(e_v[0])
            edges = list(set(edges).union(e))
        new_dict = create_file_graph(nodes, edges, df)
        s1.data = new_dict["s1"]
        s2.data = new_dict["s2"]
        n1.title.text = "Neighbors of Selected Files - " + ", ".join(selected)
        return

    s_c.data_source.on_change("selected", update_network1)

    print("ploting network 2 ...")
    s3 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s4 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n2 = figure(plot_width=500, plot_height=500, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Entities Involved in Selected Files",
                output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="black", source=s3, alpha=0.3)
    n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5, legend='s')
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s4)
    n2.legend.location = (0, 480)
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
            if node in selected:
                s_color[node] = df.loc[df["filename"] == node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"] == node]["sentiment"].values[0]
            if node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment"].values[0]
            elif "-Person" in node:
                s_color[node] = "#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#a1d76a"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    n1_c.data_source.on_change("selected", update_network2)

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
            f = s_source.data["f"][i]
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
            if node in selected:
                s_color[node] = df.loc[df["filename"] == node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"] == node]["sentiment"].values[0]
            if node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment"].values[0]
            elif "-Person" in node:
                s_color[node] = "#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#a1d76a"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s3.data = new_dict["s1"]
        s4.data = new_dict["s2"]
        n2.title.text = "Entities Involved in Selected Files - " + ", ".join(selected)
        return

    s_c.data_source.on_change("selected", update_network3)

    ########## search for entities #############
    button_group_e = RadioButtonGroup(labels=entity, active=0)
    text_e = TextInput(title="by Entity Names (separate by ','):", value='')

    # Person-Person network
    s5 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s6 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n3 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Person Relationships", output_backend="webgl")
    n3.multi_line('xs', 'ys', line_color="black", source=s5, alpha=0.5)
    c3 = n3.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s6, alpha=0.5, legend='s')
    n3.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s6)
    n3.legend.label_text_font_size = "7pt"

    # Person-Place network
    s7 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s8 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n4 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Place Relationships", output_backend="webgl")
    n4.multi_line('xs', 'ys', line_color="black", source=s7, alpha=0.5)
    c4 = n4.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s8, alpha=0.5, legend='s')
    n4.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s8)
    n4.legend.label_text_font_size = "7pt"

    # Person-Place network
    s9 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s10 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n5 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Place Relationships", output_backend="webgl")
    n5.multi_line('xs', 'ys', line_color="black", source=s9, alpha=0.5)
    c5 = n5.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s10, alpha=0.5, legend='s')
    n5.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s10)
    n5.legend.label_text_font_size = "7pt"

    n_sources = [[s5, s6], [s7, s8], [s9, s10]]
    networks = [n3, n4, n5]

    # option button for the relationships of entities
    r_option = ["Relationship Union", "Relationship Intersection"]
    button_group3 = RadioButtonGroup(labels=r_option, active=0, width=300)

    #callbacks for entities search result
    def text_e_update(attr,old,new):
        b = button_group_e.active
        x = entity[b]
        t_e = text_e.value
        a = button_group3.active
        o = r_option[a]
        tokens = t_e.split(",")
        tokens = [t.strip() for t in tokens]

        for s in n_sources:
            s[0].data = dict(xs=[], ys=[])
            s[1].data = dict(vx=[], vy=[], labels=[], color=[], s=[])

        s2_source.data = dict(x=[], y=[], f=[], h=[], d=[], c =[], s =[])
        s11.data = dict(xs=[], ys=[])
        s12.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n6.title.text = "Related Entities of Selected Files"
        t2_source.data = dict(f=[], h=[],s=[])

        if (o == "Relationship Intersection") & (len(tokens)>1):
            # update networks
            for i in range(3):
                # print("network " + str(i))
                y = entity[i]
                r = x + "_" + y
                e_d = e_dfs[x]
                s_d_x = s_dfs[x]
                s_d_y = s_dfs[y]
                selected = []
                s_color = {}
                senti = {}
                nodes = []
                edges = []
                r_n = []
                tag = "unchanged"
                for j in tokens:
                    e = j
                    temp_r = []
                    selected.append(e)
                    nodes.append(e)
                    if e in np.array(s_d_x["Entity"]):
                        s_color[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment Color"].values[0]
                        senti[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment"].values[0]
                    else:
                        scatter2.title.text = e + " is not found in the dataset"
                        n3.title.text = e + " is not found in the dataset"
                        n4.title.text = e + " is not found in the dataset"
                        n5.title.text = e + " is not found in the dataset"
                        n6.title.text = e + " is not found in the dataset"
                        return
                    e_v = []
                    if e in np.array(e_d[x]):
                        e_v = ast.literal_eval(e_d.loc[e_d[x] == e][r].values[0])
                    if e_v != []:
                        for edge in e_v:
                            temp_r.append(edge[1])
                            related = edge[1]
                            if related in np.array(s_d_y["Entity"]):
                                s_color[related] = s_d_y.loc[s_d_y["Entity"] == related]["Sentiment Color"].values[
                                    0]
                                senti[related] = s_d_y.loc[s_d_y["Entity"] == related]["Sentiment"].values[0]
                            else:
                                s_color[related] = "#91bfdb"
                                senti[related] = "Positive"
                        if (r_n == []) & (tag == "unchanged"):
                            r_n = temp_r
                            tag = "changed"
                        else:
                            r_n = list(set(r_n).intersection(temp_r))
                        edges = list(set(edges).union(e_v))
                nodes = list(set(nodes + r_n))
                tuples =list(itertools.combinations(nodes, 2))
                inter_edges = []
                for t in tuples:
                    if (t in edges) or ((t[1], t[0]) in edges):
                        inter_edges.append(t)
                g = create_graph(nodes, inter_edges, s_color, senti)
                n_sources[i][0].data = g["s1"]
                n_sources[i][1].data = g["s2"]
                networks[i].title.text = r + " Relationships for " + ", ".join(selected)

            # update timeline plot
            e_df = e_dfs[x]
            dates = []
            y_n = []
            headlines = []
            datestr = []
            senti = []
            s_color = []
            selected = []
            f_list = []
            for t in tokens:
                e = t
                selected.append(e)
                f_v = ast.literal_eval(e_df.loc[e_df[x] == e]["Documents"].values[0])
                f_list.append(f_v)
            files = f_list[0]
            for f in f_list:
                files = list(set(files).intersection(f))
            if files == []:
                scatter2.title.text = "No files are found to containing all the entities: " + ", ".join(tokens)
                s2_source.data = dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[])
                return
            else:
                for f in files:
                    v_d = df.loc[df["filename"] == f]["date"].values
                    datestr.append(v_d[0])
                    dates.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
                    v = df.loc[df["filename"] == f]["sentiment_score"].values
                    y_n.append(float(v[0]))
                    v_h = df.loc[df["filename"] == f]["headlines"].values
                    headlines.append(v_h[0])
                    v_s = df.loc[df["filename"] == f]["sentiment"].values
                    senti.append(v_s[0])
                    v_c = df.loc[df["filename"] == f]["sentiment_color"].values
                    s_color.append(v_c[0])
                s2_source.data = dict(x=dates, y=y_n, f=files, h=headlines, dstr=datestr, c=s_color, s=senti)
                scatter2.title.text ="Timeline for " + x + "s: " + ", ".join(selected)

        else:
            # update networks
            for i in range(3):
                # print("network " + str(i))
                y = entity[i]
                r = x + "_" + y
                e_d = e_dfs[x]
                s_d_x = s_dfs[x]
                s_d_y = s_dfs[y]
                selected = []
                s_color = {}
                senti = {}
                nodes = []
                edges = []
                for j in tokens:
                    e = j
                    selected.append(e)
                    nodes.append(e)
                    if e in np.array(s_d_x["Entity"]):
                        s_color[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment Color"].values[0]
                        senti[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment"].values[0]
                    else:
                        scatter2.title.text = e + " is not found in the dataset"
                        n3.title.text = e + " is not found in the dataset"
                        n4.title.text = e + " is not found in the dataset"
                        n5.title.text = e + " is not found in the dataset"
                        n6.title.text = e + " is not found in the dataset"
                        return
                    e_v = []
                    if e in np.array(e_d[x]):
                        e_v = ast.literal_eval(e_d.loc[e_d[x] == e][r].values[0])
                    if e_v != []:
                        for edge in e_v:
                            related = edge[1]
                            nodes.append(related)
                            if related in np.array(s_d_y["Entity"]):
                                s_color[related] = s_d_y.loc[s_d_y["Entity"] == related]["Sentiment Color"].values[0]
                                senti[related] = s_d_y.loc[s_d_y["Entity"] == related]["Sentiment"].values[0]
                            else:
                                s_color[related] = "#91bfdb"
                                senti[related] = "Positive"
                        edges = list(set(edges).union(e_v))
                g = create_graph(nodes, edges, s_color, senti)
                n_sources[i][0].data = g["s1"]
                n_sources[i][1].data = g["s2"]
                networks[i].title.text = r +" Relationships for " + ", ".join(selected)

            # update timeline plot
            e_df = e_dfs[x]
            dates = []
            y_n = []
            headlines = []
            datestr = []
            senti = []
            s_color = []
            selected = []
            files = []
            for t in tokens:
                e = t
                selected.append(e)
                f_v = ast.literal_eval(e_df.loc[e_df[x] == e]["Documents"].values[0])
                for f in f_v:
                    files.append(f)
                    v_d = df.loc[df["filename"] == f]["date"].values
                    datestr.append(v_d[0])
                    dates.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
                    v = df.loc[df["filename"] == f]["sentiment_score"].values
                    y_n.append(float(v[0]))
                    v_h = df.loc[df["filename"] == f]["headlines"].values
                    headlines.append(v_h[0])
                    v_s = df.loc[df["filename"] == f]["sentiment"].values
                    senti.append(v_s[0])
                    v_c = df.loc[df["filename"] == f]["sentiment_color"].values
                    s_color.append(v_c[0])
            s2_source.data = dict(x=dates, y=y_n, f=files, h=headlines, dstr=datestr, c=s_color, s=senti)
            scatter2.title.text = "Timeline for " + x + "s: " + ", ".join(selected)

    text_e.on_change("value", text_e_update)
    button_group_e.on_change("active", text_e_update)
    button_group3.on_change("active", text_e_update)

    s2_source = ColumnDataSource(data=dict(x=[], y=[], f=[], dstr=[], h=[], s=[], c=[]))
    scatter2 = figure(plot_width=550, plot_height=450, title="Timeline for Searched Entities",
                      x_axis_type='datetime', x_axis_label='Time', y_axis_label='Sentiment Score',
                      output_backend="webgl", tools="pan,box_zoom,tap,box_select,reset,save")
    s2_c = scatter2.circle('x', 'y', source=s2_source, size=7, color='c', alpha=0.8, legend='s')
    hover3 = HoverTool(
        tooltips=[
            ("file", "@f"),
            ("date", "@dstr"),
            ("headline", "@h"),
            ("sentiment", "@s"),
        ]
    )
    scatter2.add_tools(hover3)
    scatter2.x_range = Range1d(datetime.strptime('12/01/2001', "%m/%d/%Y"), datetime.strptime('01/31/2005', "%m/%d/%Y"))
    scatter2.legend.location = "top_left"
    s2_c.selection_glyph = Circle(fill_color='c', line_color="#d73027")
    s2_c.nonselection_glyph = Circle(fill_color=None, line_color="c")

    url = "http://www-edlab.cs.umass.edu/~jiesong/OntologyDrivenVisualization/data/nlp/@f"
    taptool2 = scatter2.select(type=TapTool)
    taptool2.callback = OpenURL(url=url)

    # create interactive network graph for timeline plot
    s11 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s12 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n6 = figure(plot_width=500, plot_height=500, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Related Entities of Selected Files",
                output_backend="webgl")
    n6.multi_line('xs', 'ys', line_color="black", source=s11, alpha=0.3)
    c6 = n6.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s12, alpha=0.5, legend='s')
    n6.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s12)
    n6.legend.label_text_font_size = "7pt"
    n6.legend.orientation = "horizontal"
    n6.legend.location = (10, 480)

    # update network graph when the files are selected
    def update_file_network(attr, old, new):
        inds = new['1d']['indices']

        s11.data = dict(xs=[], ys=[])
        s12.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n6.title.text = "Related Entities of Selected Files"

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
            f = s2_source.data["f"][i]
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
            if node in selected:
                s_color[node] = df.loc[df["filename"] == node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"] == node]["sentiment"].values[0]
            if node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"] == node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"] == node]["Sentiment"].values[0]
            elif "-Person" in node:
                s_color[node] = "#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#a1d76a"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s11.data = new_dict["s1"]
        s12.data = new_dict["s2"]
        n6.title.text = "Related Entities of Selected Files: " + ", ".join(selected)
        return

    s2_c.data_source.on_change("selected", update_file_network)

    t2_source = ColumnDataSource(data=dict(f=[], h=[],s=[]))
    columns2 = [
        TableColumn(field='f', title="Filename"),
        TableColumn(field='h', title="Headline"),
        TableColumn(field='s', title="Sentiment"),
    ]
    data_table2 = DataTable(source=t2_source, columns=columns2, width=450, height=400)

    # update data table 2
    def datatable2_update(attr, old, new):
        inds = new['1d']['indices']
        t1_source.data = dict(f=[], h=[],s=[])
        if inds == []:
            return
        selected = [s2_source.data["f"][i] for i in inds]
        h = [s2_source.data["h"][i] for i in inds]
        s = [s2_source.data["s"][i] for i in inds]
        t2_source.data = dict(f=selected, h=h,s=s)
        return

    s2_c.data_source.on_change("selected", datatable2_update)

    wc2_source = ColumnDataSource(data=dict(x=[], y=[], w=[], s=[], c=[], a=[]))
    wordcloud2 = figure(plot_width=700, plot_height=400, title='Word Cloud for Selected Files', x_axis_type=None,
                        y_axis_type=None)
    wordcloud2.text(x='x', y='y', text='w', text_font_size='s', text_color='c', text_font="Arial", angle='a',
                        source=wc2_source)

    # callbacks for world cloud when file is selected in scatter plot
    def wc2_s_update(attr, old, new):
        inds = new['1d']['indices']

        if inds == []:
            wc2_source.data = dict(x=[], y=[], w=[], s=[], c=[], a=[])
            wordcloud2.title.text = 'Word Cloud for Selected File'
        elif len(inds) == 1:
            i = inds[0]
            f = s2_source.data["f"][i]
            d = path.dirname(__file__)
            text = open(path.join(d, "data/nlp/" + f)).read()
            wordcloud = WordCloud(background_color="white", width=500, height=300).generate(text)
            wc_layout = wordcloud.layout_
            words = [x[0][0] for x in wc_layout]
            w_size = [str(x[1]) + "px" for x in wc_layout]
            w_color = [x[4] for x in wc_layout]
            w_x = [int(x[2][1] * wordcloud.scale) for x in wc_layout]
            w_y = [int(x[2][0] * wordcloud.scale) for x in wc_layout]
            w_a = []
            for x in wc_layout:
                if x[3] == None:
                    w_a.append(0)
                else:
                    w_a.append(np.pi / 2)
            wc2_source.data = dict(x=w_x, y=w_y, w=words, s=w_size, c=w_color, a=w_a)
            wordcloud2.title.text = "Word Cloud for Selected File: " + f
        return

    s2_c.data_source.on_change("selected", wc2_s_update)

    # callbacks for world cloud when file is selected in scatter plot
    def wc2_t_update(attr, old, new):
        inds = new['1d']['indices']
        wc2_source.data = dict(x=[], y=[], w=[], s=[], c=[], a=[])
        wordcloud2.title.text = 'Word Cloud for Selected File'

        if len(inds) == 1:
            i = inds[0]
            f = t2_source.data["f"][i]
            d = path.dirname(__file__)
            text = open(path.join(d, "data/nlp/" + f)).read()
            wordcloud = WordCloud(background_color="white", width=500, height=300).generate(text)
            wc_layout = wordcloud.layout_
            words = [x[0][0] for x in wc_layout]
            w_size = [str(x[1]) + "px" for x in wc_layout]
            w_color = [x[4] for x in wc_layout]
            w_x = [int(x[2][1] * wordcloud.scale) for x in wc_layout]
            w_y = [int(x[2][0] * wordcloud.scale) for x in wc_layout]
            w_a = []
            for x in wc_layout:
                if x[3] == None:
                    w_a.append(0)
                else:
                    w_a.append(np.pi / 2)
            wc2_source.data = dict(x=w_x, y=w_y, w=words, s=w_size, c=w_color, a=w_a)
            wordcloud2.title.text = "Word Cloud for Selected File: " + f
        return

    data_table2.source.on_change("selected", wc2_t_update)

    print("Done")

    w1 = widgetbox(button_group_f,select_f,text_f,select_k,text_k)
    l = column(row(w1,scatter),row(data_table1,wordcloud1),row(n1,n2),row(button_group_e,text_e, button_group3),row(n3,n4,n5),row(scatter2,n6),row(data_table2,wordcloud2))
    show(l)
    curdoc().add_root(l)

main()