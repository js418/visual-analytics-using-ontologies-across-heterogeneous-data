'''
Radviz for cooccurence.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_entities.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_entities

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


def main():

    entity = ["Person", "Place", "Org"]
    per_df = pd.read_csv("Person.csv")
    pla_df = pd.read_csv("Place.csv")
    org_df = pd.read_csv("Org.csv")
    s_per_df = pd.read_csv("data/entity_sentiment_avgValues/PersonSentiment.csv")
    s_pla_df = pd.read_csv("data/entity_sentiment_avgValues/PlaceSentiment.csv")
    s_org_df = pd.read_csv("data/entity_sentiment_avgValues/OrganizationSentiment.csv")
    e_dfs = {"Person": per_df, "Place":pla_df, "Org":org_df}
    s_dfs = {"Person": s_per_df, "Place": s_pla_df, "Org": s_org_df}
    r = 300
    margin = 80

    df = pd.read_csv("nlp_entities.csv")

    sentiment = ["all", "Negative", "Positive","Neutral"]
    button_group = RadioButtonGroup(labels=sentiment, active=0)
    v_select = Select(title="Radviz views:", value="Person", options=entity)
    a_select = Select(title="Radviz anchors:", value="Place", options=entity)

    with open("Radviz_entities_avg.json") as json_file:
        sources = json.load(json_file)
    json_file.close()

    ############# Radviz #############
    print("ploting Radviz ...")
    a_source = ColumnDataSource(data=sources["Person_Place"]["all"]["anchors"])
    v_source = ColumnDataSource(data=sources["Person_Place"]["all"]["views"])

    f = figure(plot_width=500, plot_height=500, title="Radviz, (views: Persons, anchors: Places), Centroid Measure",
               x_axis_type=None, y_axis_type=None, x_range=[-margin, 2 * r + margin], y_range=[-margin, 2 * r + margin],
               output_backend="webgl", tools="pan,box_zoom,box_select,reset,save")
    f.circle(r, r, radius=r, line_color="black", fill_color=None)
    c1 = f.square('a_x', 'a_y', size=8, fill_color="#7fbf7b", alpha=0.8,source=a_source)
    c2 = f.circle('v_x', 'v_y', size=6, color='c', alpha=0.4, source=v_source,legend='l')
    c2.selection_glyph = Circle(fill_color="c", line_color="#d73027")
    c2.nonselection_glyph = Circle(fill_color=None, line_color="c")

    hover1 = HoverTool(tooltips=[("anchors - " + a_select.value +": ", "@anchors")], renderers=[c1])
    hover2 = HoverTool(tooltips=[("views -  " + v_select.value + ": ", "@views")], renderers=[c2])

    f.add_tools(hover1)
    f.add_tools(hover2)

    #update data table
    def update(attr, old, new):
        inds = new['1d']['indices']
        t_source.data = dict(p=[])
        for s in n_sources:
            s[0].data = dict(xs=[], ys=[])
            s[1].data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        s_source.data = dict(x=[], y=[], f=[], h=[], d=[], c=[], s=[])
        s7.data = dict(xs=[], ys=[])
        s8.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n1.title.text = "Entity_Person Relationships"
        n2.title.text = "Entity_Place Relationships"
        n3.title.text = "Entity_Org Relationships"
        scatter.title.text = "Timeline plot for selected entity"
        n4.title.text = "Related Entities of Selected Files"
        if inds == []:
            return

        selected = []
        for i in inds:
            person = v_source.data["views"][i]
            selected.append(person)

        t_source.data = dict(p=selected)
        return

    c2.data_source.on_change("selected", update)

    ############# update the views and anchors for Radviz #############

    def radviz_update(attr, old, new):
        b = button_group.active
        s = sentiment[b]
        x = v_select.value
        y = a_select.value
        a_source.data = sources[x + "_" + y][s]["anchors"]
        v_source.data = sources[x + "_" + y][s]["views"]
        f.title.text = "Radviz, (views: " + x + " anchors: "+ y + "), Centroid Measure"
        return

    button_group.on_change("active", radviz_update)
    v_select.on_change("value", radviz_update)
    a_select.on_change("value", radviz_update)

    ############# data table #############

    t_source = ColumnDataSource(data=dict(p=[]))
    columns = [
        TableColumn(field="p", title="Entity")
    ]
    data_table = DataTable(source=t_source, columns=columns, width=400, height=400)

    #Person-Person network
    s1 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s2 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[],s=[]))
    n1 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Person Relationships",output_backend="webgl")
    n1.multi_line('xs', 'ys', line_color="black", source=s1, alpha=0.5)
    c3 = n1.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s2, alpha=0.5,legend='s')
    n1.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s2)
    n1.legend.label_text_font_size = "7pt"

    # Person-Place network
    s3 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s4 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n2 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Place Relationships", output_backend="webgl")
    n2.multi_line('xs', 'ys', line_color="black", source=s3, alpha=0.5)
    c4 = n2.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s4, alpha=0.5,legend='s')
    n2.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s4)
    n2.legend.label_text_font_size = "7pt"

    # Person-Place network
    s5 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s6 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[], s=[]))
    n3 = figure(plot_width=400, plot_height=400, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save",
                title="Person_Org Relationships", output_backend="webgl")
    n3.multi_line('xs', 'ys', line_color="black", source=s5, alpha=0.5)
    c5 = n3.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s6, alpha=0.5,legend='s')
    n3.text('vx', 'vy', text='labels', text_color="black", text_font_size="8px", text_align="center",
            text_baseline="middle", source=s6)
    n3.legend.label_text_font_size = "7pt"

    n_sources = [[s1, s2], [s3, s4], [s5, s6]]
    networks = [n1, n2, n3]

    # callback for the data table
    def table_update(attr, old,new):
        inds = new['1d']['indices']
        x = v_select.value

        for s in n_sources:
            s[0].data = dict(xs=[], ys=[])
            s[1].data = dict(vx=[], vy=[], labels=[], color=[], s=[])

        s_source.data = dict(x=[], y=[], f=[], h=[], d=[], c =[], s =[])
        s7.data = dict(xs=[], ys=[])
        s8.data = dict(vx=[], vy=[], labels=[], color=[], s=[])

        if inds == []:
            for s in n_sources:
                s[0].data = dict(xs=[], ys=[])
                s[1].data = dict(vx=[], vy=[], labels=[], color=[], s=[])
            s_source.data = dict(x=[], y=[], f=[], h=[], d=[], c=[], s=[])
            s7.data = dict(xs=[], ys=[])
            s8.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
            n1.title.text = "Entity_Person Relationships"
            n2.title.text = "Entity_Place Relationships"
            n3.title.text = "Entity_Org Relationships"
            scatter.title.text = "Timeline plot for selected entity"
            n4.title.text = "Related Entities of Selected Files"
            return

        #update networks
        for i in range(3):
            #print("network " + str(i))
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
            for j in inds:
                e = t_source.data['p'][j]
                selected.append(e)
                nodes.append(e)
                if e in np.array(s_d_x["Entity"]):
                    s_color[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment Color"].values[0]
                    senti[e] = s_d_x.loc[s_d_x["Entity"] == e]["Sentiment"].values[0]
                else:
                    s_color[e] = "#fc8d59"
                    senti[e] = "#fc8d59"
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
                            s_color[related] = "#fc8d59"
                            senti[related] = "#fc8d59"
                    edges = list(set(edges).union(e_v))
            g = create_graph(nodes, edges, s_color,senti)
            n_sources[i][0].data = g["s1"]
            n_sources[i][1].data = g["s2"]
            networks[i].title.text = r + " Relationships for " + ", ".join(selected)

        #update timeline plot
        e_df = e_dfs[x]
        dates = []
        y_n = []
        headlines = []
        datestr = []
        senti = []
        s_color = []
        selected = []
        files = []
        for k in inds:
            e = t_source.data['p'][k]
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
        s_source.data = dict(x=dates, y=y_n, f=files, h=headlines, d=datestr, c=s_color,s = senti)
        scatter.title.text = "Timeline for '" + "', '".join(selected) + "'"

        return

    data_table.source.on_change("selected", table_update)

    s_source = ColumnDataSource(data=dict(x=[], y=[], f=[], h=[], d=[], c =[], s =[]))
    TOOLS = "box_select,pan,box_zoom,reset,save"
    scatter = figure(tools=TOOLS, plot_width=600, plot_height=450, title="Timeline plot for selected entity",
                x_axis_type='datetime', x_axis_label='Time',y_axis_label="File Sentiment Score",
                output_backend="webgl")
    c6 = scatter.circle('x', 'y', source=s_source, size=10, color='c', alpha=0.8,legend='s')
    hover3 = HoverTool(
        tooltips=[
            ("document: ", "@f"),
            ("date", "@d"),
            ("headline: ", "@h")
        ]
    )
    scatter.add_tools(hover3)
    scatter.x_range = Range1d(datetime.strptime('01/01/2002', "%m/%d/%Y"), datetime.strptime('12/31/2004', "%m/%d/%Y"))
    scatter.legend.location = "top_left"

    # create interactive network graph for timeline plot
    s7 = ColumnDataSource(data=dict(xs=[], ys=[]))
    s8 = ColumnDataSource(data=dict(vx=[], vy=[], labels=[], color=[],s=[]))
    n4 = figure(plot_width=500, plot_height=450, x_axis_type=None, y_axis_type=None,
                outline_line_color=None, tools="pan,box_zoom,reset,save", title="Related Entities of Selected Files",
                output_backend="webgl")
    n4.multi_line('xs', 'ys', line_color="black", source=s7, alpha=0.3)
    c7 = n4.circle('vx', 'vy', size=20, line_color="black", fill_color='color', source=s8, alpha=0.5,legend= 's')
    n4.text('vx', 'vy', text='labels', text_color="black", text_font_size="10px", text_align="center",
            text_baseline="middle", source=s8)
    n4.legend.label_text_font_size = "7pt"
    n4.legend.orientation = "horizontal"
    n4.legend.location = (10,0)



    # update network graph when the files are selected
    def update_file_network(attr, old, new):
        inds = new['1d']['indices']

        s7.data = dict(xs=[], ys=[])
        s8.data = dict(vx=[], vy=[], labels=[], color=[], s=[])
        n4.title.text = "Related Entities of Selected Files"

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
            if node in r_per:
                s_color[node] = s_per_df.loc[s_per_df["Entity"] == node]["Sentiment Color"].values[0]
                senti[node] =s_per_df.loc[s_per_df["Entity"]== node]["Sentiment"].values[0]
            elif node in r_pla:
                s_color[node] = s_pla_df.loc[s_pla_df["Entity"]== node]["Sentiment Color"].values[0]
                senti[node] = s_pla_df.loc[s_pla_df["Entity"]== node]["Sentiment"].values[0]
            elif node in r_org:
                s_color[node] = s_org_df.loc[s_org_df["Entity"]== node]["Sentiment Color"].values[0]
                senti[node] = s_org_df.loc[s_org_df["Entity"]== node]["Sentiment"].values[0]
            elif node in selected:
                s_color[node] = df.loc[df["filename"]== node]["sentiment_color"].values[0]
                senti[node] = df.loc[df["filename"]== node]["sentiment"].values[0]
            elif "-Person" in node:
                s_color[node]="#f441e8"
                senti[node] = "Person Parent Node"
            elif "-Place" in node:
                s_color[node] = "#e2f441"
                senti[node] = "Place Parent Node"
            elif "-Org" in node:
                s_color[node] = "#f49141"
                senti[node] = "Org Parent Node"

        new_dict = create_graph(nodes, edges, s_color, senti)
        s7.data = new_dict["s1"]
        s8.data = new_dict["s2"]
        n4.title.text = "Related Entities of Selected Files: " +", ".join(selected)
        return

    c6.data_source.on_change("selected", update_file_network)

    w = widgetbox(button_group,v_select,a_select)
    l = column(row(w,f,data_table),row(n1,n2,n3),row(scatter,n4))

    curdoc().add_root(l)

main()