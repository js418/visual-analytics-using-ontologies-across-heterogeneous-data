import pandas as pd
import numpy as np
import ast
import itertools
import radviz_centroid_optimization as rv
import json
from datetime import datetime

def main():

    """
    t = pd.read_csv("data/Person_Org_Cooccurence.csv")
    t = t.T
    h = t.head(1).values.tolist()[0]
    t.columns = h
    t= t[1:]
    t.to_csv("data/Org_Person_Cooccurence.csv")
    t = pd.read_csv("data/Person_Place_Cooccurence.csv")
    t = t.T
    h = t.head(1).values.tolist()[0]
    t.columns = h
    t = t[1:]
    t.to_csv("data/Place_Person_Cooccurence.csv")
    t = pd.read_csv("data/Place_Org_Cooccurence.csv")
    t = t.T
    h = t.head(1).values.tolist()[0]
    t.columns = h
    t = t[1:]
    t.to_csv("data/Org_Place_Cooccurence.csv")
    """

    df = pd.read_csv("data/file_date_entity_type.csv", encoding="ISO-8859-1")
    """
    df = pd.read_csv("data/file_date_entity_type.csv", encoding="ISO-8859-1")
    temp_d = df.groupby("entity_type")["entity"].apply(lambda x: list(set(x)))
    persons = temp_d["PERSON"]
    orgs = temp_d["ORGANIZATION"]
    places = temp_d["GPE"]
    """
    persons = []
    orgs = []
    places = []
    for i in range(len(df.index)):
        t = df.loc[i]["entity_type"]
        if t == "PERSON":
            persons.append(df.loc[i]["entity"])
        elif t == "ORGANIZATION":
            orgs.append(df.loc[i]["entity"])
        elif t == "GPE":
            places.append(df.loc[i]["entity"])
    persons=list(set(persons))
    orgs = list(set(orgs))
    places = list(set(places))
    print("entites in file_date_entity_type.csv:")
    print(len(persons), len(places), len(orgs))

    co_df = pd.read_csv("data/Org_Org_Cooccurence.csv")
    org = co_df["Unnamed: 0"].tolist()
    co_df = pd.read_csv("data/Person_Place_Cooccurence.csv")
    per = co_df["Unnamed: 0"].tolist()
    pla = co_df.columns.values.tolist()
    pla.pop(0)
    persons = list(set(persons)& set(per))
    places = list(set(places)& set(pla))
    orgs = list(set(orgs)& set(org))
    print("after intersection with cooccurrence file:")
    print(len(persons), len(places), len(orgs))

    s_dfs = {}
    s_df = pd.read_csv("data/entity_sentiment_ratio/PersonSentiment.csv")
    s_dfs["Person"] = s_df
    s_per = s_df["Entity"].tolist()
    persons = list(set(persons).intersection(s_per))
    s_df = pd.read_csv("data/entity_sentiment_ratio/PlaceSentiment.csv")
    s_dfs["Place"] = s_df
    s_pla = s_df["Entity"].tolist()
    places = list(set(places).intersection(s_pla))
    s_df = pd.read_csv("data/entity_sentiment_ratio/OrganizationSentiment.csv")
    s_dfs["Org"] = s_df
    s_org = s_df["Entity"].tolist()
    orgs = list(set(orgs).intersection(s_org))
    print("after intersection with sentiment file:")
    print(len(persons), len(places), len(orgs))

    entity = ["Person", "Place", "Org"]
    e_lists = {"Person": persons, "Place": places, "Org": orgs}
    co_lists = {"Person": per, "Place": pla, "Org": org}
    co_dfs = {}
    colors = {}
    labels = {}
    index = {}
    for x in entity:
        s_df = s_dfs[x]
        e_colors = []
        e_labels = []
        e_index = []
        for e in e_lists[x]:
            c_v = s_df.loc[s_df["Entity"] == e]["Sentiment Color"].values
            e_colors.append(c_v[0])
            s_v = s_df.loc[s_df["Entity"] == e]["Sentiment"].values
            e_labels.append(s_v[0])
            e_index.append(co_lists[x].index(e))
        colors[x] = np.array(e_colors)
        labels[x] = np.array(e_labels)
        index[x] = np.array(e_index)

    sources = {}
    out_file = open("Radviz_entities_ratio.json", "w")
    r = 300
    for x in entity:
        for y in entity:
            print("save data for " + x + "_" + y)
            sources[x + "_" + y] ={}
            d = pd.read_csv("data/" + x + "_" + y + "_Cooccurence.csv")
            row_index = index[x]
            col_index = index[y] + 1
            new_df = d.iloc[row_index, col_index]
            co_dfs[x + "_" + y] = new_df

            g = rv.radviz_optimization(new_df.values)
            g.optimize()
            view = g.get_view()
            anchor = g.get_anchors()

            anchor = r * anchor + r
            view = r * view + r

            a = dict(a_x=anchor[:, 0].tolist(), a_y=anchor[:, 1].tolist(), anchors=e_lists[y])
            v = dict(v_x=view[:, 0].tolist(), v_y=view[:, 1].tolist(), views=e_lists[x], c=colors[x].tolist(), l=labels[x].tolist())
            sources[x + "_" + y]["all"] = {"anchors":a, "views":v}

            s_color = {"Negative": "#fc8d59", "Positive":"#91bfdb", "Neutral":"#ffffbf"}
            for a in ["Negative", "Positive", "Neutral"]:
                s_df = s_dfs[x]
                s_e = s_df[s_df["Sentiment"] == a]["Entity"].values
                l = list(set(e_lists[x]).intersection(s_e))
                if l == []:
                    n_a = dict(a_x=[], a_y=[], anchors=[])
                    n_v = dict(v_x=[], v_y=[], views=[], c=[],l=[])
                else:
                    row_index = []
                    for b in l:
                        row_index.append(co_lists[x].index(b))
                    s_new_df = d.loc[row_index, e_lists[y]]

                    n_g = rv.radviz_optimization(s_new_df.values)
                    n_g.optimize()
                    n_view = n_g.get_view()
                    n_anchor = n_g.get_anchors()
                    n_anchor = r * n_anchor + r
                    n_view = r * n_view + r

                    n_a = dict(a_x=n_anchor[:, 0].tolist(), a_y=n_anchor[:, 1].tolist(), anchors=e_lists[y])
                    n_v = dict(v_x=n_view[:, 0].tolist(), v_y=n_view[:, 1].tolist(), views=l, c=[s_color[a]] * len(l),
                               l=[a] * len(l))

                sources[x + "_" + y][a] = {"anchors": n_a, "views": n_v}

    json.dump(sources, out_file, indent=4)
    out_file.close()


    """
    df = pd.read_csv("nlp_entities.csv")
    h_d = pd.read_csv("data/headlines.csv")
    h_headlines = np.array(h_d["headline"]).tolist()
    h_files = np.array(h_d["filename"]).tolist()

    senti = []
    s_score = []
    s_color = []
    #d = []
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
        #d.append(datetime.strptime(v_d[0], "%m/%d/%Y"))
        v_h = df.loc[df["filename"] == f]["headlines"].values
        e_headlines.append(v_h[0])

    # h_d.drop(labels="headline", axis=1, inplace=True)
    # h_d.drop(labels="filename", axis=1, inplace=True)
    h_d.drop(labels="Unnamed: 0", axis=1, inplace=True)
    h_d.drop(labels="Unnamed: 0.1", axis=1, inplace=True)
    h_d.insert(len(h_d.columns), "sentiment", senti)
    h_d.insert(len(h_d.columns), "sentiment_score", s_score)
    h_d.insert(len(h_d.columns), "sentiment_color", s_color)
    h_d.insert(len(h_d.columns), "e_headlines", e_headlines)
    #h_d.insert(len(h_d.columns), "file_date", d)
    h_d.insert(len(h_d.columns), "datestr", datestr)

    N = len(h_d.columns)
    new_d = h_d.iloc[:, 0:(N - 7)]

    words = new_d.columns.values.tolist()

    print("data processing ...")
    g = rv.radviz_optimization(new_d.values)
    g.optimize()
    view = g.get_view()
    anchor = g.get_anchors()

    r = 300
    anchor = r * anchor + r
    view = r * view + r
    margin = 80

    all_a = dict(a_x=anchor[:, 0].tolist(), a_y=anchor[:, 1].tolist(), anchors=words)
    all_v = dict(v_x=view[:, 0].tolist(), v_y=view[:, 1].tolist(), views=h_files,
                 h=h_headlines, c=s_color, score=s_score, senti=senti,
                 dstr=datestr, e_h=e_headlines)
    sources = {}
    out_file = open("Radviz_files.json", "w")
    for x in ["Negative", "Positive", "Neutral"]:
        print("save date for " + x)
        data = h_d[h_d["sentiment"] == x]
        if len(data.index) == 0:
            a = dict(a_x=[], a_y=[], anchors=[])
            v = dict(v_x=[], v_y=[], views=[], h=[], c=[],
                     score=[], senti=[], dstr=[], e_h=[])
        else:
            n_d = data.iloc[:, 0:(N - 7)]
            n_g = rv.radviz_optimization(n_d.values)
            n_g.optimize()
            n_view = n_g.get_view()
            n_anchor = n_g.get_anchors()
            n_anchor = r * n_anchor + r
            n_view = r * n_view + r

            a = dict(a_x=n_anchor[:, 0].tolist(), a_y=n_anchor[:, 1].tolist(), anchors=words)
            v = dict(v_x=n_view[:, 0].tolist(), v_y=n_view[:, 1].tolist(), views=np.array(data["filename"]).tolist(),
                     h=np.array(data["headline"]).tolist(), c=np.array(data["sentiment_color"]).tolist(),
                     score=np.array(data["sentiment_score"]).tolist(), senti=np.array(data["sentiment"]).tolist(),
                     dstr=np.array(data["datestr"]).tolist(), e_h=np.array(data["e_headlines"]).tolist())

        sources[x] = {"anchors": a, "views": v}
    sources["all"] = {"anchors":all_a, "views":all_v}

    json.dump(sources, out_file, indent=4)
    out_file.close()
    """

main()