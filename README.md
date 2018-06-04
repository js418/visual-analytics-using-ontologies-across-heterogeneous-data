# OntologyDrivenVisualization
A collaborative effort between researchers at Umass Amherst under the guidance of Professor Grinstein to incorporate Machine Learning into the visualizaiton pipeline.

File descriptions:

********** .py files ************

basic_viz.py: an example for basic scatter plots and network graph

radviz_centroid_optimization.py: updated version from Julian's radviz_centroid_optimization.py

radviz_temp.py: an example for Radviz by Bokeh

nlp_date_preprocessing.py: data processing for NLP results

nlp_viz_part1.py: interactive visualizations for NLP results -- scatter plots and network graphs for the information of each file.

nlp_viz_part2.py: entity search -- search an entity by index or name and show the relationships with other entities.

nlp_viz_part3.py: Radviz for cooccurrence

nlp_viz_files.py: interactive Radviz for sentiment files

nlp_viz_entities.py: interactive Radviz for sentiment entities

********** .csv files ************

original_entities.csv: all the entities for original database

nlp_entities.csv: entities for NLP results

Person.csv: variables for PERSON entity

Org.csv: variables for ORGANIZATION entity

Place.csv:variables for PLACE entity

********** Bokeh Server ************

command: $ bokeh serve XXX.py

local server:localhost:5006/
