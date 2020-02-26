from sklearn.externals import joblib
import cv2
import numpy as np
import pandas as pd

from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

# model=joblib.load('knn_model.pkl')
import glob
def ExtractHOG(img):
    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, multichannel=True)
    return ftr
  
def preprocessing_part_two(arr):
    arr_feature=[]
    # for i in range(np.shape(arr)):
    #     arr_feature.append(ExtractHOG(arr[i]))
    arr_feature.append(ExtractHOG(arr))
    return arr_feature
# data_test_ftr=cv2.imread(r"C:\Users\HP\Desktop\ISO  HACK\api\Fruit_Recognition\FruitsDB\Oranges\Test\test.jpg")
# data_test_ftr=np.asarray(data_test_ftr)
# y_knn_pred = model.predict(preprocessing_part_two(data_test_ftr))

# strr=""
# # file_=glob.glob(r"C:\Users\HP\Desktop\ISO  HACK\api\Fruit_Recognition\FruitsDB\Oranges\Test\test.jpg")
# img=cv2.imread(r"C:\Users\HP\Desktop\ISO  HACK\mango.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img=resize(img, (72, 72),anti_aliasing=True)
# # data_test_p=preprocessing_part_two(img)
# data_test_ftr= preprocessing_part_two(img)
# y_knn_pred = model.predict(data_test_ftr)
# print(y_knn_pred)