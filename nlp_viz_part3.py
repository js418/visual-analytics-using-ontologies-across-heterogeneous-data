'''
Radviz for cooccurence.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve nlp_viz_part3.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/nlp_viz_part3

in your browser.
'''
import pandas as pd
import numpy as np
import ast
from bokeh.models import CustomJS, ColumnDataSource,HoverTool,BoxSelectTool,TapTool,Range1d
from bokeh.layouts import row, column, widgetbox,layout
from bokeh.io import show
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput,Button, RadioButtonGroup, Select, Slider,DataTable, DateFormatter, TableColumn
import networkx as nx
from datetime import datetime
import itertools
import radviz_centroid_optimization as rv

def main():
    df = pd.read_csv("data/file_date_entity_type.csv", encoding="ISO-8859-1")
    temp_d = df.groupby("entity_type")["entity"].apply(lambda x: list(set(x)))
    persons = temp_d["PERSON"]
    #orgs = temp_d["ORGANIZATION"]
    places = temp_d["GPE"]
    print(len(persons),len(places))


    df = pd.read_csv("data/Entity_Cooccurence_new.csv")
    entities = df["Unnamed: 0"].tolist()
    persons = list(set(persons) & set(entities))
    #orgs = list(set(orgs).intersection(entities))
    places = list(set(places)& set(entities))
    print(len(persons), len(places))

    row_index = []
    for x in persons:
        row_index.append(entities.index(x))
    col_index = []
    for y in places:
        col_index.append(entities.index(y))
    df.drop(labels="Unnamed: 0", axis=1, inplace=True)
    new_df = df.loc[row_index,places]

    #############centroid layout #############
    g = rv.radviz_optimization(new_df.values)
    g.optimize()
    view = g.get_view()
    anchor = g.get_anchors()

    r = 300
    anchor = r * anchor + r
    view = r * view + r
    margin = 80

    a_source = ColumnDataSource(data=dict(a_x= anchor[:, 0], a_y= anchor[:, 1],anchors= places))
    v_source = ColumnDataSource(data=dict(v_x= view[:, 0], v_y= view[:, 1], views= persons))
    f = figure(plot_width=500, plot_height=500, title="Persons (anchors: Places), Centroid Measure",
               x_axis_type=None, y_axis_type=None, x_range=[-margin, 2 * r + margin], y_range=[-margin, 2 * r + margin],
               output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    f.circle(r, r, radius=r, line_color="black", fill_color=None)
    c1 = f.circle('a_x', 'a_y', size=10, fill_color="#af8dc3", source= a_source)
    c2 = f.circle('v_x', 'v_y', size=10, fill_color="red", alpha= 0.4, source= v_source)

    hover1 = HoverTool(tooltips=[("anchor: ", "@anchors")], renderers=[c1])
    #hover2 = HoverTool(tooltips=[("Person: ", "@views")], renderers=[c2])

    f.add_tools(hover1)
    #f.add_tools(hover2)

    #############equant layout #############
    g2 = rv.radviz_optimization(new_df.values)
    t = g2.get_thetas()
    view2 = g2.get_view()
    anchor2_x = r * np.cos(t) + r
    anchor2_y = r * np.sin(t) + r
    view2 = r * view2 + r

    a2_source = ColumnDataSource(data=dict(a_x=anchor2_x, a_y=anchor2_y, anchors=places))
    v2_source = ColumnDataSource(data=dict(v_x=view2[:, 0], v_y=view2[:, 1], views=persons))

    f2 = figure(plot_width=500, plot_height=500, title="Persons (anchors: Places), Equant Measure",
                x_axis_type=None, y_axis_type=None, x_range=[-margin, 2 * r + margin],y_range=[-margin, 2 * r + margin],
                output_backend="webgl",tools="pan,box_zoom,box_select,reset,save")
    f2.circle(r, r, radius=r, line_color="black", fill_color=None)
    c3 = f2.circle('a_x', 'a_y', size=10, fill_color="blue", source=a2_source)
    c4 = f2.circle('v_x', 'v_y', size=10, fill_color="red", alpha=0.4, source=v2_source)

    hover3 = HoverTool(tooltips=[("anchor: ", "@anchors")], renderers=[c3])
    #hover4 = HoverTool(tooltips=[("Person: ", "@views")], renderers=[c4])

    f2.add_tools(hover3)
    #f2.add_tools(hover4)

    ############# data table #############

    t_source = ColumnDataSource(data=dict(p=[]))
    columns = [
        TableColumn(field="p", title="Person"),
    ]
    data_table = DataTable(source=t_source, columns=columns, width=400, height=400)

    def update(attr,old,new):
        inds = new['1d']['indices']
        if inds == []:
            t_source.data = dict(p=[])
            return

        selected = []
        r_index = []
        for i in inds:
            person = v2_source.data["views"][i]
            r_index.append(entities.index(person))
            selected.append(person)

        t_source.data = dict(p=selected)

        """
        new_df = df.loc[r_index, places]
        #print(new_df.values)
        g = rv.radviz_optimization(new_df.values)
        g.optimize()
        view = g.get_view()
        anchor = g.get_anchors()

        anchor = r * anchor + r
        view = r * view + r
        #print(len(view),len(selected))

        a_source.data =dict(a_x=anchor[:, 0], a_y=anchor[:, 1], anchors=places)
        v_source.data=dict(v_x=view[:, 0], v_y=view[:, 1], views=selected)
        """
        return

    c4.data_source.on_change("selected", update)

    l = column(row(f2,data_table),f)
    show(l)

    curdoc().add_root(l)


main()