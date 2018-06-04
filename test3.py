import bokeh
from bokeh.io import output_notebook
from bokeh.layouts import layout
from bokeh.models import Label
from bokeh.plotting import figure
from bokeh.plotting import show
import csv
from numpy import interp
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Dropdown
from bokeh.layouts import row, column
from bokeh.models.widgets import Select
from bokeh.plotting import curdoc, figure
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bokeh import mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import Counter
filePath = "C:/grinstein/Kriti_WordCloud/OntologyDrivenVisualization-kriti_nlp-vis/data/filename_keywords_rake.csv"
def updateFigure():
    filename = str(selectFile.value)
    allData = []
    data = []
    with open(filePath, 'r') as csvfile:
        next(csvfile)
        allData = csv.reader(csvfile, delimiter=',')
        ##Making word cloud for the selected file
        data = [row for row in allData if row[1] == filename]
        print (data)
        words = []
        for row in data:
            words.append(row[2])
        print (words[0])
    csvfile.close()
    wordcloud = WordCloud(background_color="white",max_font_size=40).generate(words[0])
    plt.imshow(wordcloud)#, interpolation="bilinear"
    plt.axis("off")
    plt.show()
    fig = mpl.to_bokeh(plt.figure())
    return fig
filenames = []
with open(filePath, 'r') as file:
    next(file)
    allData = csv.reader(file, delimiter=',')
    for row in allData:
        if row[1] not in filenames:
            filenames.append(row[1])
def changeSelectedFile(attr,old,new):
    layout.children[1] = updateFigure()
selectFile = Select(title="Select document:", value="1101163308494.txt", options=filenames)
selectFile.on_change('value', changeSelectedFile)
layout = column(selectFile, updateFigure())
curdoc().add_root(layout)