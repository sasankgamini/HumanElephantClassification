#importing modules/dependencies
import numpy as np
# import matplotlib.pyplot as plt
# import keras
# from keras.datasets import mnist
import cv2

# ((trainimgs, trainlabels),(testimgs,testlabels))= mnist.load_data()
# class_names=[0,1,2,3,4,5,6,7,8,9]

# print(trainimgs[0])
# plt.imshow(trainimgs[0])
# plt.show()

# cv2.imshow('first', trainimgs[0])
# cv2.waitKey()
# cv2.destroyAllWindows()

# # Multiple images
# for n in range(0,25,1):
#     plt.subplot(5,5,n+1)
#     plt.imshow(trainimgs[n])
# plt.show()
    
# #building the model(blueprint)
# model=keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128,activation='relu'), #activation if it passes certian threshold
#     keras.layers.Dense(10,activation='softmax') #gives percentages for each number in third layer
#     ])

# #Compile the model/properties of model(giving extra features/personalizing)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics= ['accuracy'])

# #Train the model(Construction of the house)
# model.fit(trainimgs,trainlabels,epochs=5)

# model.save("handwrittendigitrecogmodel.h5")


from keras.models import load_model
model = load_model("handwrittendigitrecogmodel.h5")

image = cv2.imread('/Users/sasankgamini/Desktop/MachineLearningProjects/HumanElephantClassification/41yjh1X2ahL._AC_.jpg')
image = cv2.resize(image,(800,600))
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,bw = cv2.threshold(grayimage,90,255,cv2.THRESH_BINARY_INV)
contours, hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,255,255),3)
rectangles = []
for n in contours:
    print(cv2.boundingRect(n))
    rectangles.append(cv2.boundingRect(n))

for rect in rectangles:
        cv2.rectangle(image, (rect[0]-18, rect[1]-18), (rect[0] + rect[2]+18, rect[1] + rect[3]+18), (0, 255, 0), 3)
        roi = bw[rect[1]-18:rect[1]+rect[3]+18, rect[0]-18:rect[0]+rect[2]+18]
        if roi.any():
                roi = cv2.resize(roi, (28, 28), image, interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                image1 = np.reshape(roi,(1,28,28))
                print(image1.shape)
                predictions = model.predict(image1)
                print(predictions)
                print(str(np.argmax(predictions[0])))
                #Put text on the screen
                cv2.putText(image, str(np.argmax(predictions[0])), (rect[0], rect[1]+20),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)


cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()


 
