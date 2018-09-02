

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def train_and_report_logistic(train_X, train_ids,test_X,test_ids):
    model = LogisticRegression(solver = 'lbfgs')
    model.fit(train_X, train_ids)
    # Returns a NumPy Array
    # Predict for One Observation (image)
    predicted = model.predict(test_X)
    # generate evaluation metrics
    print(metrics.accuracy_score(test_ids, predicted))
    return model

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Dropout , BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import  SGD
from sklearn import preprocessing

def get_onehot_encoded(class_ids):
    assert len(np.unique(class_ids))>1  , "You need more than one class in the class label" 
    lb = preprocessing.LabelBinarizer()
    train_lbl = lb.fit_transform(class_ids)
    return train_lbl,lb.classes_

# define baseline model
def create_model(n_classes,input_dim,init_mode='truncated_normal'):
	# create model
	model = Sequential()
	model.add(Dense(128, input_dim=input_dim, kernel_initializer=init_mode,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Dense(n_classes, activation='softmax'))
	# Compile model
	sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
	return model

def train_and_report_mlp_keras(train_X,train_ids,test_X,test_ids,epochs=1, batch_size=32):
    test_lbl,_ = get_onehot_encoded(test_ids)
    train_lbl,_ = get_onehot_encoded(train_ids)
    n_classes,input_dim = train_lbl.shape[1],train_X.shape[1]
    model = create_model(n_classes,input_dim)
    model.fit(train_X, train_lbl,epochs=epochs,batch_size=batch_size)
    score = model.evaluate(test_X, test_lbl, batch_size=batch_size)
    print("[INFO] your final reported test accuracy score is {}% on {} samples having {} classes ". \
          format(round((score[1]*100),2),len(test_X),test_lbl.shape[1]))
    return model

def train_and_report(train_X, train_ids,test_X,test_ids,algo_name,epochs=10, batch_size=32):
    supported_algos = ['logistic','mlp']
    assert algo_name in supported_algos," wrong param passed , pls pass {} ".format(supported_algos)
    if algo_name=='logistic':
        return train_and_report_logistic(train_X, train_ids,test_X,test_ids)
    if algo_name=='mlp':
        return train_and_report_mlp_keras(train_X,train_ids,test_X,test_ids,epochs, batch_size)
    
    