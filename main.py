import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
from sklearn.cluster import KMeans


#Columns --> age,sex,bmi,children,smoker,region,charges

# creating a data frame
insuranceData = pd.read_csv("insurance.csv") #original data
insuranceDataLabels = pd.read_csv("insuranceLabels.csv", header = None) #labels for data
insuranceDataBinning = pd.read_csv("insuranceBinning.csv", header = None) #binning data
insuranceNumeric = pd.read_csv("insuranceNumeric.csv") #numeric data

# this function makes files for different data tecniques(FOR EASIER READABILITY)
def makeLabels():
    fileData = []
    with open("insurance.csv", "r") as f:
        fileLines = f.readlines()
        for line in fileLines:
            fileData.append(line.split(","))

    with open("insuranceLabels.csv", "w") as fp:
        for linee in fileData:
            fp.write("age="+linee[0] + ",sex="+linee[1] + ",bmi="+linee[2] + ",children="+linee[3] + ",smoker="+linee[4] + ",region="+linee[5] + ",charges="+linee[6] )
    

#making bins to store numerial data and view easier(FOR ASSOCIATION RULES)
def makeBins():
    fileData = []
    with open("insurance.csv", "r") as f:
        fileLines = f.readlines()
        for line in fileLines:
            fileData.append(line.split(","))

    with open("insuranceBinning.csv", "w") as fp:

        for linee in fileData:
            age = ""
            bmi = ""
            charges = ""
            if int(linee[0]) < 28:
                age = "age=<28"
            elif int(linee[0]) >= 28 and int(linee[0]) <= 37:
                age = "age=28-37"
            elif int(linee[0]) >= 38 and int(linee[0]) <= 47:
                age = "age=38-47"
            elif int(linee[0]) >= 48 and int(linee[0]) <= 57:
                age = "age=48-57"
            elif int(linee[0]) > 57:
                age = "age=57<"
            
            if float(linee[2]) < 24:
                bmi = ",bmi=<24"
            elif float(linee[2]) >= 24 and float(linee[2]) <= 32:
                bmi = ",bmi=24-32"
            elif float(linee[2]) >= 33 and float(linee[2]) <= 41:
                bmi = ",bmi=33-41"
            elif float(linee[2]) >= 42 and float(linee[2]) <= 50:
                bmi = ",bmi=42-50"
            elif float(linee[2]) > 50:
                bmi = ",bmi=50<"

            if float(linee[6]) < 11621:
                charges = ",charges=<11621"
            elif float(linee[6]) >= 11621 and float(linee[6]) <= 22122:
                charges = ",charges=11621-22122"
            elif float(linee[6]) >= 22123 and float(linee[6]) <= 32623:
                charges = ",charges=22123-32623"
            elif float(linee[6]) >= 32624 and float(linee[6]) <= 43124:
                charges = ",charges=32624-43124"
            elif float(linee[6]) >= 43125 and float(linee[6]) <= 53625:
                charges = ",charges=43125-53625"
            elif float(linee[6]) > 53625:
                charges = ",charges=53625<"

            fp.write(age + ",sex="+linee[1] + bmi+ ",children="+linee[3] + ",smoker="+linee[4] + ",region="+linee[5] + charges +'\n')
    

#this function creates numeric values for the data(FOR CLUSTERING)
def makeNumeric():
    fileData = []
    with open("insurance.csv", "r") as f:
        fileLines = f.readlines()
        for line in fileLines:
            fileData.append(line.split(","))

    with open("insuranceNumeric.csv", "w") as fp:
        for linee in fileData:
            sex =   ""
            smoker = ""
            region = ""

            if linee[1] == "female":
                sex = "0"
            else:
                sex = "1"
            
            if linee[4] == "yes":
                smoker = "1"
            else:
                smoker = "0"

            if linee[5] == "southeast":
                region = "1"
            elif linee[5] == "southwest":
                region = "2"
            elif linee[5] == "northeast":
                region = "3"
            elif linee[5] == "northwest":
                region = "4"

            fp.write(linee[0] + "," + sex + "," + linee[2] + "," + linee[3] + "," + smoker + "," + region + "," + linee[6] )


#this function creates histograms for the columns
def createCharts():
    #age chart
    age = insuranceData['age']
    plt.hist(age, bins = 5, color = 'green', edgecolor = 'black')
    plt.title("Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

    #sex chart
    sex = insuranceData['sex']
    plt.hist(sex, bins = 2, color = 'green', edgecolor = 'black')
    plt.title("Sex")
    plt.xlabel("Sex")
    plt.ylabel("Frequency")
    plt.show()

    #bmi chart
    bmi = insuranceData['bmi']
    plt.hist(bmi, bins = 5, color = 'green', edgecolor = 'black')
    plt.title("BMI")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.show()

    #children chart
    children = insuranceData['children']
    plt.hist(children, bins = 6, color = 'green', edgecolor = 'black')
    plt.title("Children")
    plt.xlabel("Children")
    plt.ylabel("Frequency")
    plt.show()

    #smoker chart
    smoker = insuranceData['smoker']
    plt.hist(smoker, bins = 2, color = 'green', edgecolor = 'black')
    plt.title("Smoker")
    plt.xlabel("Smoker")
    plt.ylabel("Frequency")
    plt.show()

    #region chart
    region = insuranceData['region']
    plt.hist(region, bins = 4, color = 'green', edgecolor = 'black')
    plt.title("Region")
    plt.xlabel("Region")
    plt.ylabel("Frequency")
    plt.show()

    #charges chart
    charges = insuranceData['charges']
    plt.hist(charges, bins = 6, color = 'green', edgecolor = 'black')
    plt.title("Charges")
    plt.xlabel("Charges")
    plt.ylabel("Frequency")
    plt.show()


#this function prints all association rules for common trends in the set
def assosiationRules():
    insuranceRecords = []
    for i in range(0, 1338):
        insuranceRecords.append([str(insuranceDataBinning.values[i,j]) for j in range(0, 7)])

    association_rules = apriori(insuranceRecords, min_support=0.05, min_confidence=0.7, min_lift=3, min_length=2)
    association_results = list(association_rules)

    for x in association_results:
        pair = x[0]
        items = [y for y in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(x[1]))
        print("Confidence: " + str(x[2][0][2]))
        print("Lift: " + str(x[2][0][3]))
        print("=====================================")


#this function is for finding kmeans clusters each column with charges
#----takes in cluster total and column name----
def clustering(clusterValue, columnValue):
    c1 = insuranceNumeric[[columnValue, 'charges']]
    cluster1 = KMeans(n_clusters=clusterValue)
    cluster1.fit(c1)
    c1['cluster'] = cluster1.fit_predict(c1)
    plt.scatter(c1[columnValue], c1['charges'], c=c1['cluster'], cmap='rainbow')
    plt.xlabel(columnValue)
    plt.ylabel('charges')
    plt.show()





if __name__ == '__main__':
    #assosiationRules()
    #makeBins()
    #createCharts()
    clustering(3, 'region')
    #makeNumeric()