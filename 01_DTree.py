import csv
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


def reader_csv():
    data = open(r"AllElectronics.csv","rt")
    reader = csv.reader(data)
    header = reader.__next__()
    return header,reader

def getfeature_label(header,data,featurelist,lablelist):
    for row in data:
        lablelist.append(row[-1])
        line = {}
        for i in range(len(row)-1):
            line[header[i]]=row[i]
        featurelist.append(line)
    return featurelist,lablelist

def transform(featurelist,lablelist):
    vec = DictVectorizer()
    labvac = preprocessing.LabelBinarizer()
    featurearr = vec.fit_transform(featurelist).toarray()
    labarr = labvac.fit_transform(lablelist)
    return featurearr,labarr

if __name__ == '__main__':
    header,data = reader_csv()
    featurelist = []
    lablelist = []
    featurelist,lablelist =getfeature_label(header,data,featurelist,lablelist)
    feature_arr, lab_arr = transform(featurelist,lablelist)
    print(feature_arr, lab_arr )

