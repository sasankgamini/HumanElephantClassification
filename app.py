import pandas as pd
file = pd.read_csv("ElephantHumanDataset.txt")
# print(file["Height"])
inputData = file.drop("Class",1)
outputData = file["Class"]
print(outputData)