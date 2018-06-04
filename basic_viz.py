import os
import csv
import datetime
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.plotting import figure
import networkx as nx

def loopFiles(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(os.path.join(directory, filename), 'r') as readFile:
            continue
            # Do NLP operations here and remove continue.
    return


def loadEntities(directory, tabData):

    with open(os.path.join(directory, "Entities_DAT.txt"), 'r') as readFile:
        reader = csv.reader(readFile, delimiter='\t')
        for line in reader:
            if "/" in line[0]:
                d = datetime.datetime.strptime(line[0], "%m/%d/%Y")
                for i in range(3,len(line)):
                    tabData[line[i]] = dict(date = d, persons = [], locations = [], orgs = [])
    readFile.close

    #manully add 3 text files
    tabData["1101163556822"]= dict(date = datetime.datetime.strptime("12/23/2003", "%m/%d/%Y"), persons = [], locations = [], orgs = [])
    tabData["1101163685855"]= dict(date = datetime.datetime.strptime("2/13/2003", "%m/%d/%Y"), persons = [], locations = [], orgs = [])
    tabData["1101163356450"]= dict(date = datetime.datetime.strptime("5/5/2002", "%m/%d/%Y"), persons = [], locations = [], orgs = [])


    with open(os.path.join(directory, "Entities_PER.txt"), 'r') as readFile:
        reader = csv.reader(readFile, delimiter='\t')
        for line in reader:
            for i in range(3,len(line)):
                tabData[line[i]]["persons"].append(line[0])
    readFile.close

    with open(os.path.join(directory, "Entities_LOC.txt"), 'r') as readFile:
        reader = csv.reader(readFile, delimiter='\t')
        for line in reader:
            for i in range(3,len(line)):
                tabData[line[i]]["locations"].append(line[0])
    readFile.close

    with open(os.path.join(directory, "Entities_ORG.txt"), 'r') as readFile:
        reader = csv.reader(readFile, delimiter='\t')
        for line in reader:
            for i in range(3,len(line)):
                tabData[line[i]]["orgs"].append(line[0])
    readFile.close

    return

# several layouts: 'circular_layout','random_layout','shell_layout','spring_layout','spectral_layout',
def graph_draw(g, layout=nx.spring_layout, node_color="white", text_color="black"):
    pos = layout(g)
    labels = [ str(v) for v in g.nodes() ]
    vx, vy = zip(*[ pos[v] for v in g.nodes() ])
    #print(vx)
    xs, ys = [], []
    for (a, b) in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xs.append([x0, x1])
        ys.append([y0, y1])
    #print(xs)
    f = figure(plot_width=1000, plot_height=1000,
               x_axis_type=None, y_axis_type=None,
               outline_line_color=None,
               tools="pan,box_zoom,reset,save",output_backend="webgl")
    f.multi_line(xs, ys, line_color="blue")
    f.circle(vx, vy, size=16, line_color="black", fill_color=node_color)
    f.text(vx, vy, text=labels, text_color=text_color, text_font_size="8px", text_align="center", text_baseline="middle")
    return f

def main():
    visualizationGroup = True
    tabData = {}
    dataDirectory = os.getcwd() + "\data\\"

    if visualizationGroup:
        dataDirectory += "visualization\\"
        loadEntities(dataDirectory, tabData)
    else:
        dataDirectory += "nlp\\"
        loopFiles(dataDirectory)

    #print(tabData)

    text = []
    date = []
    dateStr = []
    personNum = []
    locNum = []
    orgNum = []
    for file in tabData:
        text.append(file)
        v = tabData[file]
        date.append(v["date"])
        dateStr.append(v["date"].strftime("%Y-%m-%d"))
        personNum.append(len(v["persons"]))
        locNum.append(len(v["locations"]))
        orgNum.append(len(v["orgs"]))

    source = ColumnDataSource(data=dict(x=date, y0=personNum, y1=locNum, y2=orgNum, dateString = dateStr,textFile = text))

    hover1 = HoverTool(
        tooltips=[
            ("text file", "@textFile"),
            ("date", "@dateString"),
            ("persons", "@y0"),
        ]
    )

    hover2 = HoverTool(
        tooltips=[
            ("text file", "@textFile"),
            ("date", "@dateString"),
            ("locations", "@y1"),
        ]
    )

    hover3 = HoverTool(
        tooltips=[
            ("text file", "@textFile"),
            ("date", "@dateString"),
            ("organizations", "@y2"),
        ]
    )

    TOOLS = "box_select,pan,box_zoom,crosshair,undo,redo,reset,help"

    f1 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of persons involved in textfile",
                x_axis_type='datetime', x_axis_label='Time', y_axis_label='number of persons')
    f1.circle('x', 'y0', source=source, color="navy",alpha=0.5)
    f1.add_tools(hover1)

    f2 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of locations involved in textfile",
                x_range=f1.x_range,y_range=f1.y_range, x_axis_type='datetime', x_axis_label='Time', y_axis_label='number of locations')
    f2.triangle('x', 'y1', source=source,color="firebrick",alpha=0.5)
    f2.add_tools(hover2)

    f3 = figure(tools=TOOLS, plot_width=400, plot_height=400, title="The number of organizations involved in textfile",
                x_range=f1.x_range,y_range=f1.y_range, x_axis_type='datetime', x_axis_label='Time', y_axis_label='number of locations')
    f3.square('x', 'y2', source=source, color="olive",alpha=0.5)
    f3.add_tools(hover3)

    plots_1 = gridplot([[f1,f2,f3]])
    output_file("initial_viz.html")
    show(plots_1)

    p_list = ["Rex Luthor","Bruce Rinz","Laurel Sulfate","Delwin Sanderson","VonRyker", "Philip Boynton","Tom Seeger", "John Torch"]
    #p_list = ["Laurel Sulfate"]
    PV = []
    PE = []
    for file in tabData:
        p = tabData[file]["persons"]
        if (len(p) > 1) and (list(set(p_list).intersection(p)) != []):
        #if (len(p) > 1):
            PV = list(set(PV).union(p))
            unique_pairs = [(p[p1], p[p2]) for p1 in range(len(p)) for p2 in range(p1 + 1, len(p))]
            PE = list(set(PE).union(unique_pairs))

    g = nx.Graph()
    g.add_nodes_from(PV)
    g.add_edges_from(PE)

    f = graph_draw(g, node_color="red", text_color="black")

    show(f)


if __name__ == '__main__':
    main()
