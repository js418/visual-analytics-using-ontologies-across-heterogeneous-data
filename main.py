import os
import csv


def loopFiles(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(os.path.join(directory, filename), 'r') as readFile:
            continue
            # Do NLP operations here and remove continue.
    return


def loadEntities(directory, tabData):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(os.path.join(directory, filename), 'r') as readFile:
            reader = csv.reader(readFile, delimiter='\t')
            for line in reader:
                tabData.append(line)
    return


def main():
    visualizationGroup = True
    tabData = []
    dataDirectory = os.getcwd() + "\data\\"

    if visualizationGroup:
        dataDirectory += "visualization\\"
        loadEntities(dataDirectory, tabData)
    else:
        dataDirectory += "nlp\\"
        loopFiles(dataDirectory)

    print(tabData)


if __name__ == '__main__':
    main()