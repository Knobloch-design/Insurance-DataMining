import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Columns --> age,sex,bmi,children,smoker,region,charges

# creating a data frame
insuranceData = pd.read_csv("insurance.csv") #original data
insuranceDataLabels = pd.read_csv("insuranceLabels.csv", header = None) #labels for data
insuranceDataBinning = pd.read_csv("insuranceBinning.csv", header = None) #binning data
insuranceNumeric = pd.read_csv("insuranceNumeric.csv") #numeric data
insuranceDataBinning2 = pd.read_csv("insuranceBinning2.csv") #binning data for averages of charges

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
    

#making bins to store numerial data and view easier
def makeBins():
    fileData = []
    with open("insurance.csv", "r") as f:
        fileLines = f.readlines()
        for line in fileLines:
            fileData.append(line.split(","))

    with open("insuranceBinning2.csv", "w") as fp:

        for linee in fileData:
            age = ""
            bmi = ""
            if int(linee[0]) < 28:
                age = "<28"
            elif int(linee[0]) >= 28 and int(linee[0]) <= 37:
                age = "28-37"
            elif int(linee[0]) >= 38 and int(linee[0]) <= 47:
                age = "38-47"
            elif int(linee[0]) >= 48 and int(linee[0]) <= 57:
                age = "48-57"
            elif int(linee[0]) > 57:
                age = "57<"
            
            if float(linee[2]) < 24:
                bmi = "<24"
            elif float(linee[2]) >= 24 and float(linee[2]) <= 32:
                bmi = "24-32"
            elif float(linee[2]) > 32 and float(linee[2]) <= 41:
                bmi = "32.001-41"
            elif float(linee[2]) > 41 and float(linee[2]) <= 50:
                bmi = "41.001-50"
            elif float(linee[2]) > 50:
                bmi = "50<"


            fp.write(age + ","+ linee[1] + "," + bmi+ ","+linee[3] + ","+linee[4] + ","+ linee[5] + "," + linee[6] )
    

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
    

    

    clusterGT= cluster1.fit_predict(c1)
    print(clusterGT)

    plt.scatter(c1[columnValue], c1['charges'], c=clusterGT, cmap='rainbow')
    plt.xlabel(columnValue)
    plt.ylabel('charges')
    plt.show()


    #c2 = insuranceNumeric[[columnValue, 'charges']]
    #meanS = MeanShift(bandwidth=clusterValue).fit(c2)


#this chart shows standard direct correlation between a single attribute and charge
#----takes in column name----
def regression(columnName):
    '''
    plt.scatter(insuranceData[columnName], insuranceData['charges'], color = 'lightblue', edgecolor = 'black')
    plt.title(columnName + " vs Charges")
    plt.xlabel(columnName)
    plt.ylabel("Charges")
    plt.show()
    '''
    plt.pie(insuranceDataBinning2[columnName].value_counts(), labels = insuranceDataBinning2[columnName].value_counts().index, autopct='%1.1f%%')
    plt.title(columnName)
    plt.show()

#can only use for bmi, age, children, and charges
def correlationChart():

    #print("Correlation Coefficient for " + columnName + " and charges: " + str(insuranceNumeric[columnName].corr(insuranceNumeric['charges'])))
    corr = insuranceData.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap = 'coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,7,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(insuranceData.columns)
    ax.set_yticklabels(insuranceData.columns)
    plt.show()

    """
    test = insuranceData.apply(lambda x: x.factorize()[0]).corr(method='pearson', min_periods=1)
    import seaborn as sns
    sns.heatmap(test, annot=True)
    plt.show()
    print(test)
    """


#to find average charge cost per column possibility
#----takes in column name----
def averageChargePerColumn(columnName):
    listofoptions = []
    for x in insuranceDataBinning2[columnName]:
        if x not in listofoptions:
            listofoptions.append(x)

    for x in listofoptions:
        print("Average charge for " + columnName + " " + str(x) + " is: " + str(insuranceDataBinning2[insuranceDataBinning2[columnName] == x]['charges'].mean()))


if __name__ == '__main__':
    #assosiationRules() #this function prints all association rules for common trends in the set
    #makeBins() #this function makes bins for the data
    #createCharts() #this function creates charts for the data
    #clustering(5, 'region') #this function is for finding kmeans clusters each column with charges
    #makeNumeric() #this function makes the data numeric
    #print(insuranceNumeric.describe()) #statistics of each column
    regression('bmi') #this chart shows standard direct correlation between a single attribute and charge
    #correlationChart() #this chart shows correlation between all columns
    #averageChargePerColumn('region') #to find average charge cost per column possibility