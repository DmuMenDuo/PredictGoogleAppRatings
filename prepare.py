import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def read():

    return pd.read_csv('./google-play-store-apps/googleplaystore.csv')


def removeNaData():

    data = read()
    print(data.info())
    data.dropna(axis=0, inplace=True)
    print("--------after remove na ------")
    print(data.info())
    return data

def changeSize(size):
    if 'G' in size:
        return float(size.replace('G', '')) * 1024 * 1024 * 1024
    elif 'M' in size:
        return float(size.replace('M', '')) * 1024 * 1024
    elif 'K' in size:
        return float(size.replace('K', '')) * 1024
    else:
        return 0

def changeInstalls(install):
    return install.replace(',', '').replace('+', '')

def changePrice(price):
    if '$' in price:
        return float(price.replace('$', ''))
    else:
        return 0


if __name__ == '__main__':
    data = removeNaData()
    result = data.copy()
    print(data.head())

    # preprocessing Category
    # get all type
    categoryType = data["Category"].unique()

    # change type to dict
    categoryDict = {}
    cont = 0
    for type in categoryType:
        categoryDict[type] = cont
        cont += 1

    result["Category"] = data["Category"].map(categoryDict).astype(int)
    print(result["Category"])

    # preprocessing Reviews
    result["Reviews"] = data["Reviews"].astype(int)

    # preprocessing Size
    result["Size"] = data["Size"].map(changeSize).astype('float32')


    # preprocessing Installs
    result["Installs"] = data["Installs"].map(changeInstalls).astype(int)

    # preprocessing Type
    typeType = data["Type"].unique()
    typeDict = {}
    cont = 0
    for type in typeType:
        typeDict[type] = cont
        cont += 1
    result["Type"] = data["Type"].map(typeDict).astype(int)


    # preprocessing Price
    result["Price"] = data["Price"].map(changePrice).astype('float32')


    # preprocessing Content Rating
    contentType = data["Content Rating"].unique()
    contentDict = {}
    cont = 0
    for type in contentType:
        contentDict[type] = cont
        cont += 1
    result["Content Rating"] = data["Content Rating"].map(contentDict).astype(int)


    # preprocessing Genres
    genresType = data["Genres"].unique()
    genresDict = {}
    cont = 0
    for type in genresType:
        genresDict[type] = cont
        cont += 1
    result["Genres"] = data["Genres"].map(genresDict).astype(int)

    # preprocessing Rating

    result["Rating"] = data["Rating"].astype('float32')
    print(result.info())

    list = ["App","Rating","Category","Reviews","Size",
                                "Installs","Type","Price","Content Rating","Genres"]
    nparray = result.as_matrix(list)

    x = nparray.shape[1]

    for i in range(2, x):
        print("pearson correlation coefficient: " + list[1] + " and " + list[i])
        print(pearsonr(nparray[:,1].astype(float),nparray[:,i].astype(float)))