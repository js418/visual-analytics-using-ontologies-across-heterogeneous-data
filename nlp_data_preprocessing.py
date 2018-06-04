import pandas as pd
import numpy as np
import ast

def GenerateEntityFile(entity,inputFile,inputEntity,entity_df,file_list):
    p_df = pd.read_csv(inputFile[0])
    #p_df[entity] = p_df[entity].str.lower()
    #p_df["Sorted_CondProb"] = p_df["Sorted_CondProb"].str.lower()
    p = np.array(p_df[entity].tolist())

    # map entity to the file list in all entities file
    p_file = {}
    for file in file_list:
        v = entity_df.loc[entity_df["filename"] == file][entity].values
        if v[0] != "None":
            p_list = ast.literal_eval(v[0])
            for person in p_list:
                if person in p_file:
                    if file not in p_file[person]:
                        p_file[person].append(file)
                else:
                    p_file[person] = [file]
    print(len(p_file))

    # get the file list for each variable of entity
    person_file = []
    for i in p:
        if i in p_file:
            person_file.append(p_file[i])
        else:
            person_file.append("None")
    print(len(person_file))
    p_df.insert(3, "Documents", person_file)

    # add the relationships
    for i in range(len(inputFile)):
        df = pd.read_csv(inputFile[i])
        #df[entity] = df[entity].str.lower()
        #df["Sorted_CondProb"] = df["Sorted_CondProb"].str.lower()
        r = np.array(df["Sorted_CondProb"].tolist())
        p_org = []
        p_org_w = []
        for j in range(len(r)):
            r_list = ast.literal_eval(r[j])
            relations = []
            relations_w = []
            if r_list != []:
                for k in r_list:
                    relations.append((df[entity][j], k[0]))
                    relations_w.append((df[entity][j], k[0], k[1]))
            p_org.append(relations)
            p_org_w.append(relations_w)
        p_df.insert(len(p_df.columns), entity+"_"+inputEntity[i], p_org)
        p_df.insert(len(p_df.columns), entity+"_"+inputEntity[i]+"_Weight", p_org_w)

    p_df.drop(p_df.columns[[0]], axis=1, inplace=True)
    p_df.drop(labels="Sorted_CondProb",axis=1, inplace=True)
    p_df = p_df[p_df["Documents"] != "None"]
    p_df.to_csv(entity+".csv", sep=',', encoding='utf-8')
    return

def main():

    # --------------generate NLP result entities file-----------------------
    df = pd.read_csv("data/file_date_entity_type.csv",encoding = "ISO-8859-1")
    temp_d = df.groupby(["filename","date","entity_type"])["entity"].apply(lambda x: list(set(x)))
    entity_df = temp_d.unstack()
    entity_df.to_csv("nlp_entities.csv",sep=',', encoding='utf-8')

    entity_df = pd.read_csv("nlp_entities.csv")
    c = np.array(entity_df["filename"].tolist())
    c0_df = pd.read_csv("data/wordvec_cluster0.csv", encoding="ISO-8859-1")
    c0 = np.array(c0_df["filename"].tolist())

    # add cluster label and color to each file
    c_label = []
    color_label = []
    for d in c:
        if d in c0:
            c_label.append("cluster 0")
            color_label.append("blue")
        else:
            c_label.append("cluster 1")
            color_label.append("green")
    entity_df.insert(len(entity_df.columns),"cluster_label",c_label)
    entity_df.insert(len(entity_df.columns), "color_label", color_label)
    entity_df.fillna("None", inplace=True)

    """
    # add the keyword (highest rake score) to each file
    k_df = pd.read_csv("data/filename_keywords_rake.csv")
    keywords =[]
    for file in c:
        keywords.append(k_df[k_df.filename == file]["keyword"].head(1).values[0])
    entity_df.insert(2, "keyword", keywords)
    """

    # manually set the date for 1101163556822 and 1101163356450
    entity_df.set_value(474, "date", "['12/23/2003']")
    entity_df.set_value(379, "date", "['5/5/2002']")
    for i in range(len(entity_df.index)):
        date = entity_df.loc[i]["date"]
        date = date.split("['")[1].split("']")[0]
        if "," in date:
            date = date.split("', '")[0]
        entity_df.loc[i,"date"] = date


    entity_df.rename(columns={"GPE": "Place", "ORGANIZATION": "Org", "PERSON":"Person"}, inplace=True)

    # add nodes and edges of each file for network graph
    nodes = []
    edges = []
    for file in c:

        v = entity_df.loc[entity_df["filename"] == file]["Person"].values
        if v[0] != "None":
            person = ast.literal_eval(v[0])
        else:
            person = []
        v = entity_df.loc[entity_df["filename"] == file]["Place"].values
        if v[0] != "None":
            place = ast.literal_eval(v[0])
        else:
            place = []
        v = entity_df.loc[entity_df["filename"] == file]["Org"].values
        if v[0] != "None":
            org = ast.literal_eval(v[0])
        else:
            org = []

        person_e = []
        for i in person:
            person_e.append((file+"-Person",i))
        place_e = []
        for j in place:
            place_e.append((file+"-Place", j))
        org_e = []
        for k in org:
            org_e.append((file+"-Org", k))

        N = person + place + org + [file, file+"-Person", file+"-Place", file+"-Org"]
        nodes.append(N)
        E = person_e + place_e + org_e + [(file, file+"-Person"), (file, file+"-Place"), (file, file+"-Org")]
        edges.append(E)
    entity_df.insert(len(entity_df.columns),"all_nodes", nodes)
    entity_df.insert(len(entity_df.columns), "all_edges", edges)

    # add headlines for each file
    h_df = pd.read_csv("data/doc_sentiment_verbs_adjs_advs_ps.csv")
    headline = []
    senti = []
    s_score = []
    s_color = []
    r_score = []
    for file in c:
        v = h_df.loc[h_df["filenames"] == file]["headlines"].values
        headline.append(v[0])
        s = h_df.loc[h_df["filenames"] == file]["Sentiment"].values
        if s[0] == 0:
            senti.append("Neutral")
            s_color.append("#ffffbf")
        elif s[0] == 1:
            senti.append("Negative")
            s_color.append("#fc8d59")
        elif s[0] == 2:
            senti.append("Positive")
            s_color.append("#91bfdb")
        r = h_df.loc[h_df["filenames"] == file]["Neg_Pos_Ratio"].values
        r_score.append(r[0])
        neg = h_df.loc[h_df["filenames"] == file]["neg_scores"].values
        pos = h_df.loc[h_df["filenames"] == file]["pos_scores"].values
        score = float(pos[0]) - float(neg[0])
        s_score.append(score)
    entity_df.insert(2, "headlines", headline)
    entity_df.insert(3, "sentiment", senti)
    entity_df.insert(4, "sentiment_score", s_score)
    entity_df.insert(5, "sentiment_color", s_color)
    entity_df.insert(6, "ratio_sentiment_score", r_score)

    # add top 10 neighbors for each file
    n_df = pd.read_csv("data/Title_nearest_neighbors.csv")
    filename = np.array(h_df["filenames"]).tolist()
    n_df.insert(0, "filename", filename)
    neighbors = []
    neighbors_edge = []
    for file in c:
        n = []
        e = []
        for i in range(1,11):
            f = "filename" + str(i)
            v = n_df.loc[n_df["filename"] == file][f].values
            n.append(v[0])
            e.append((file, v[0]))
        neighbors.append(n)
        neighbors_edge.append(e)
    entity_df.insert(len(entity_df.columns), "file_neighbors", neighbors)
    entity_df.insert(len(entity_df.columns), "file_neighbors_edge", neighbors_edge)

    entity_df.to_csv("nlp_entities.csv", sep=',', encoding='utf-8')

    # ------------generate PERSON entity file ---------------------
    inputFile = ["data/Persons_Orgs_CondProb_Person.csv", "data/Persons_Places_CondProb_Person.csv","data/Persons_Persons_CondProb_Person.csv"]
    inputEntity = ["Org", "Place","Person"]
    GenerateEntityFile("Person",inputFile,inputEntity,entity_df,c)

    # ------------generate ORGANIZATION entity file ---------------------
    inputFile = ["data/Persons_Orgs_CondProb_Org.csv", "data/Places_Orgs_CondProb_Org.csv","data/Orgs_Orgs_CondProb_Org.csv"]
    inputEntity = ["Person", "Place","Org"]
    GenerateEntityFile("Org", inputFile, inputEntity, entity_df, c)

    # ------------generate PLACE entity file ---------------------
    inputFile = ["data/Persons_Places_CondProb_Place.csv", "data/Places_Orgs_CondProb_Place.csv","data/Places_Places_CondProb_Place.csv"]
    inputEntity = ["Person", "Org","Place"]
    GenerateEntityFile("Place", inputFile, inputEntity, entity_df, c)



main()