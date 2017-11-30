import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import glob
import re

import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

GaussianNBModel = GaussianNB()
KneighboursModel_5 = make_pipeline(
    PCA(2100),
    KNeighborsClassifier(n_neighbors=5)
)
KneighboursModel_15 = make_pipeline(
    PCA(2100),
    KNeighborsClassifier(n_neighbors=15)
)
KneighboursModelWithScaler_5 = make_pipeline(
    StandardScaler(),
    PCA(2100),
    KNeighborsClassifier(n_neighbors=5)
)
KneighboursModelWithScaler_15 = make_pipeline(
    StandardScaler(),
    PCA(2100),
    KNeighborsClassifier(n_neighbors=15)
)
SVCModelRBF = SVC(kernel='rbf', C=5, gamma=5)

SVCModelLinear = SVC(kernel='linear', C=1e-2)

SVCModelWithScalerRBF = make_pipeline(
    StandardScaler(),
    PCA(2100),
    SVC(kernel='rbf', C=5, gamma=5)
)
SVCModelWithScalerLinear = make_pipeline(
    StandardScaler(),
    PCA(2100),
    SVC(kernel='linear', C=1e-2)
)
NeuralModel = MLPClassifier(solver='lbfgs',
                            hidden_layer_sizes=(5, 4),
                            activation='identity',
                            random_state=0
                            )


def SeqBug(input):
    return np.array(list(input), dtype=np.float)


def LoadGreyImage(inPut):
    data = Image.open(inPut)  # .convert("L")

    return data


#
# def LoadImage(inPut):
#     data = Image.open(inPut)  # .convert("L")
#     data.load()
#     imageData = np.asarray(data, dtype="float")
#     return imageData
def LoadImage(inPut):
    data = Image.open(inPut)  # .convert("L")
    temp = data.resize((64, 48), Image.ANTIALIAS)
    temp.load()
    imageData = np.asarray(temp, dtype="float")
    return imageData / 255


def imageDate(i):
    Reg = r"katkam-(\d\d\d\d)(\d\d)(\d\d)(\d\d)"
    m = re.search(Reg, i)
    if m:
        time = '' + m.group(1) + '-' + m.group(2) + '-' + m.group(3) + ' ' + m.group(4) + ':00'
        return time
    else:
        return None


def getGreyValue(img):
    pixel = []
    for x in range(0, 64):
        for y in range(0, 48):
            pixel_value = img.getpixel((x, y))
            pixel.append(pixel_value)
    avg = sum(pixel) / len(pixel)
    cp = []
    for px in pixel:
        if px > avg:
            cp.append(1)
        else:
            cp.append(0)
    return cp


def classfiy_aHash(image1, size=(64, 48)):
    image1 = image1.resize(size).convert('L').filter(ImageFilter.BLUR)
    image1 = ImageOps.equalize(image1)
    code1 = getGreyValue(image1)
    return code1


def imageDateBack(i):
    Reg = r"(\d\d\d\d)-(\d\d)-(\d\d)\s(\d\d)\:(\d\d)"
    m = re.search(Reg, i)
    if m:
        time = '' + m.group(1) + m.group(2) + m.group(3) + m.group(4) + m.group(5) + '00'
        return time
    else:
        return None


def LoadWeatherDataFrame(path):
    df = pd.read_csv(path, skiprows=16, error_bad_lines=False)
    return df


def renaming(a, b):
    i = 0
    for filename in glob.glob('katkam-scaled/*.jpg'):
        #     print(filename)
        for s in a:
            if s in filename:
                newName = b[i] + str(i) + '.jpg'
                os.rename(filename, newName)
                i += 1
            else:
                continue


def PrintResults(Xtrain, ytrain, xtest, ytest):
    models = [GaussianNBModel,
              KneighboursModel_5,
              KneighboursModel_15,
              KneighboursModelWithScaler_5,
              KneighboursModelWithScaler_15,
              SVCModelRBF,
              SVCModelLinear,
              SVCModelWithScalerRBF,
              SVCModelWithScalerLinear,
              NeuralModel
              ]
    # fit each model
    for i, m in enumerate(models):
        m.fit(Xtrain, ytrain)
    modelName = [' GaussianNBModel',
                 '  KneighboursModel_5',
                 '  KneighboursModel_15',
                 '  KneighboursModelWithScaler_5',
                 '  KneighboursModelWithScaler_15',
                 '   SVCModelRBF',
                 '   SVCModelLinear',
                 '   SVCModelWithScalerRBF',
                 '   SVCModelWithScalerLinear',
                 '    NeuralModel'
                 ]
    # print the score for each model
    for i, m in enumerate(models):
        temp = m.score(xtest, ytest)
        print(modelName[i] + "'s score:" + str(temp))

print(' -------------------------------------------------------------------')
print('|This program NORMALIZE the weather for increasing accuracy purpose!|')
print(' -------------------------------------------------------------------')
AllGreyImageArray = []
AllColourImageArray = []
AllImageDateArray = []
for filename in glob.glob('katkam-scaled/*.jpg'):
    AllImageDateArray.append(imageDate(filename))
    AllGreyImageArray.append(classfiy_aHash(LoadGreyImage(filename)))
    AllColourImageArray.append(LoadImage(filename))
WeatherDataFrame = pd.DataFrame()
frames = []
for fileName in glob.glob('yvr-weather/*.csv'):
    df = LoadWeatherDataFrame(fileName)
    frames.append(df)
DataFrameOfDate = pd.DataFrame({'ImageDate': AllImageDateArray, 'Grey': AllGreyImageArray})
AllCsvDataFrame = (pd.concat(frames, ignore_index=True)).dropna(subset=['Weather'])
'''
Grey Image Prediction with Weather label
'''
# result.to_csv('lol.csv')
''' Gray method preparation '''
df1 = pd.DataFrame({'date': AllImageDateArray})
df2 = pd.DataFrame(AllGreyImageArray)
AppandedDF = pd.concat([df1, df2], axis=1, join='inner')
DateCleaned = pd.concat([AllCsvDataFrame.set_index('Date/Time'),
                         AppandedDF.set_index('date')],
                        axis=1, join='inner').reset_index()
X = DateCleaned.iloc[:, 25:].values
y = DateCleaned['Weather'].values
'''train gray image data set'''
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('\nUsing Grey image to predict Weather:')
PrintResults(SeqBug(X_train), y_train, SeqBug(X_test), y_test)

'''
#this renaming function should be commended out when we are not dealing with deep learning.
DF = pd.DataFrame()
DF['date'] = DateCleaned['index']
DF['weather'] = DateCleaned['Weather']
DF['DateInString'] = DF['date'].apply(imageDateBack)
FileName = DF['DateInString'].values
Classifier = DF['weather'].values
renaming(FileName, Classifier)
'''

'''
Grey Image prediction with Time
'''
print('\nUsing Grey image to predict Time:')
y_time = DateCleaned['Time'].values
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X, y_time)
PrintResults(SeqBug(X_train_time), y_train_time, SeqBug(X_test_time), y_test_time)
''' Colour method preparation '''
df1Colour = pd.DataFrame({'date': AllImageDateArray})
ColourArray = np.asarray(AllColourImageArray)
df2Colour = pd.DataFrame(ColourArray.reshape(ColourArray.shape[0],
                                             ColourArray.shape[1] * ColourArray.shape[2] * ColourArray.shape[3]))
AppandedDFColour = pd.concat([df1Colour, df2Colour], axis=1, join='inner')
DateCleanedColour = pd.concat([AllCsvDataFrame.set_index('Date/Time'),
                               AppandedDFColour.set_index('date')],
                              axis=1, join='inner').reset_index()
XColour = DateCleanedColour.iloc[:, 25:]
yColour = DateCleaned['Weather']
'''train colour image data set'''
X_train_Colour, X_test_Colour, y_train_Colour, y_test_Colour = train_test_split(XColour, yColour)
print('\nUsing Colourful image to predict Weather:')
PrintResults(X_train_Colour, y_train_Colour, X_test_Colour, y_test_Colour)
print('\nUsing Colourful image to predict Time:')
y_time = DateCleaned['Time'].values
X_train_Colour, X_test_Colour, y_train_Colour, y_test_Colour = train_test_split(XColour, y_time)
PrintResults(X_train_Colour, y_train_Colour, X_test_Colour, y_test_Colour)
'''
Create different model
'''
print(' ---')
print('|EOF|')
print(' ---')

