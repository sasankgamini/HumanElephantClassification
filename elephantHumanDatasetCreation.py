#human height = 3 - 7 ft
#human weight = 30lbs - 300lbs

#elephant height = 6ft - 15ft
#elephant weight = 300lbs - 1000lbs

import random

file = open('ElephantHumanDataset.txt','w')

for i in range(0,1000,1):
    file.write(str(random.uniform(3,7))+","+str(random.uniform(30,300))+",human\n")

for i in range(0,1000,1):
    file.write(str(random.uniform(6,10))+","+str(random.uniform(300,1000))+",elephant\n")


file.close()
