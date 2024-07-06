#!/usr/bin/env python
# coding: utf-8

# ### 1. Create a s3 bucket
# 
# Using boto 3 -> its a AWS SDK for python which helps to crate,update and delete AWS resources form our python scripts
# 
# Boto3 makes it easy to integrate your python application,library or script with aws services including aws s3,EC2,dynamodb..

# In[6]:


import boto3
s3 = boto3.resource('s3')


# In[7]:


#creating a bucket
bucket_name = "yahoofinancestpricesagemaker"
try:
    s3.create_bucket(Bucket=bucket_name)
    print("s3 bucket has been created succesfully")
except Exception as e:
    print('S3 error:', e)


# ### 2. Create train and validation csv

# In[18]:


get_ipython().system('pip install yfinance')


# In[8]:


#extracting data about apple stock from 2023 to 2024
import pandas as pd
from datetime import datetime
import yfinance as yf

#initialize parameters
start_date = datetime(2019,1,1)
end_date = datetime(2021,1,1)

df_data = yf.download('AAPL',start=start_date,end=end_date)

df_data.reset_index(inplace=True)

#creating a dataframe
df_data


# ### 3. Extract load and transform the data

# In[9]:


#dropping  Adj CLose and Date columns

df_data.drop(axis=1, columns=['Adj Close'],inplace=True)
df_data.drop(axis=1, columns=['Date'],inplace=True)


# In[10]:


#Extracting features

#getiing all the features
df_data_features = df_data.iloc[:-1 , :]
df_data_features


# In[11]:


#now we will rename teh open price as targets
#here we are taking from the 2nd row
df_data_targets = df_data.iloc[1:, 0].rename("Targets")
df_data_targets


# In[12]:


#now combinng the Target adn the dataframe

df_data_features['Target'] = list(df_data_targets)

first_column = df_data_features.pop('Target')
df_data_features.insert(0,'Target',first_column)

df_data_final = df_data_features
df_data_final


# In[13]:


#note for xgboost to work we need to assign target as the first columns


# ### Train test split

# In[14]:


import numpy as np
df_randomized = df_data_final.sample(frac=1, random_state=123)
df_randomized

#makinf the data ina random order for random selection


# In[15]:


#splitting data basedon randomised data

train_data, test_data = np.split(df_randomized,[int(0.8*len(df_randomized))])
print(train_data.shape, test_data.shape)


# In[9]:


### set path & upload dataset to s3

import os
prefix= 'xgboost-as-a-built-in-algo'

train_csv_path = 's3://{}/{}/{}/{}'.format(bucket_name,prefix,'train','train.csv')
test_csv_path = 's3://{}/{}/{}/{}'.format(bucket_name,prefix,'test','test.csv')
print(train_csv_path)
print(test_csv_path)


# In[18]:


train_data.to_csv(train_csv_path, index=False,header=False)
test_data.to_csv(test_csv_path, index=False,header=False)


# ### 4. Build XGBoost Model

# ### How to use SageMaker XGBoost
# 
# we can use it as a framework and a built in algorithm
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
# 
# we'll be using the XGboost as a built inalgorithm

# In[2]:


import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput


# ### Find a xgboost image url and build a xgboost container

# In[3]:


xgboost_container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1.7-1")
display(xgboost_container)


# ## Initialize hyperparameters
# ### Booster parameters
# 
# 
#         - max_depth: This parameter controls the maximum depth of a tree. Increasing this value makes 
#                      the model more complex and  more likely to overfit. It's crucial for controlling 
#                      the complexity of the model.
# 
#         - eta (learning rate): It controls the step size shrinkage used to prevent overfitting. After each 
#                                boosting step, the eta parameter shrinks the feature weights to make the boosting
#                                process more conservative. Lower values make the model more robust by shrinking 
#                                the contribution of each tree.
# 
#         - gamma: This parameter specifies the minimum loss reduction required to make a further partition on 
#                  a leaf node of the tree. It acts as a regularization parameter, controlling the complexity 
#                  of the tree model.
# 
#         - min_child_weight: It defines the minimum sum of instance weight (hessian) needed in a child. 
#                             In other words, it helps control over-fitting. Higher values prevent a model 
#                             from learning relations which might be highly specific to the particular 
#                             sample selected for a tree.
# 
#         - subsample: Denotes the fraction of observations to be randomly sampled for each tree.
#                      Lower values make the algorithm more conservative and prevents overfitting
#                      but setting it too low might lead to underfitting.
# 
# These parameters collectively help in fine-tuning the performance and generalization capability of the XGBoost model by controlling its complexity, learning aggressiveness, and robustness against noise in the training data. Adjusting these parameters requires balancing between model performance and overfitting, often through cross-validation or other validation techniques.

# In[4]:


#settingthe hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"reg:squarederror",
        "early_stopping_rounds":10,
        "num_round":1000}


# ### setting an output path where the trained model will be saved

# In[10]:


output_path = 's3://{}/{}/{}/'.format(bucket_name,prefix,'output')
print(output_path)


# ## construct a sagemaker estimator that calles the xgboost-container
# 
#     Enable the train_use_spot_instances constructor arg - a simple self-explanatory boolean
#     
#     set the train_max_wait constructor arg -this is an int arg representing the amount of time you are willing to wait for         the spot infrastructure to become available.Some instances types are harder to get at spot prices and you may have to wait 
#     longer.you are not charged for the time spent once the spot instance have been successfully procured
#     
#     train_max_run - The timeout in seconds for training

# In[11]:


#setting estimator with hyperparameters
estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                          hyperparameters = hyperparameters,
                                          role = sagemaker.get_execution_role(),
                                          instance_count=1,
                                          instance_type='ml.m4.xlarge',
                                          volume_size =5, #5 GB,
                                          output_path=output_path,
                                          use_spot_instances=True,
                                          max_run = 300,
                                          max_wait = 600)


# ## Define the data type and paths to the training and validation datasets

# In[12]:


content_type = "csv"
train_input = TrainingInput("s3://{}/{}/{}/".format(bucket_name,prefix,'train'), content_type=content_type)
test_input = TrainingInput("s3://{}/{}/{}/".format(bucket_name,prefix,'test'), content_type=content_type)


# ### Execute the XGBoost training job

# In[13]:


estimator.fit({'train':train_input,'validation':test_input})


# In[14]:


#using spot intance helps to save cost


# # 5. Deploy trained xgb model as endpoint
# 
#     1. Environment
#        within sagemaker -serilization by User
#        ##outside sagemaker  - serialization by Endpoint ***
#    
#     2. Method to invoke the endpoint
#        ## API-single prediction
#        s3 bucket -batch prediction
#     
#     3.Datatype based on method
#        ## API - JSON
#        s3 BUcket -csv

# Using the json msg we'll trigger the json msg send to api gateway which 
# via a lambda function will trigger the api endpoint
# 
# now that here the absed on the type of application it can be within or outside the sagemaker application
# 
# methods to invoke endpoints include single and batch prediction here we are using single prediction thus we make use of api
# 
# 

#  NOw in order to host a model through AWS EC2 using amazon sagemaker,deploy the model that you trained and run a training job ny calling the #"deploy method" of the xgb_model_estimator
# 
# when you call the deploy method few key things that you need to specify
# 
#     Initial_instance_count(int)-- The number of instances to deploy the model
# 
#     instance_typr(str) - The type of instances that you want to operate your deployed model
# 
#     serializer(int) -Serialize the input data of various formats( a numpy array,list,file or buffer) to a csv formatted  string.we use this because the xgboost algorithm accepts the input files in csv format

# In[15]:


from sagemaker.serializers import CSVSerializer
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge',serializer=CSVSerializer())


# In[16]:


xgb_predictor.endpoint_name


# ### Make predictions with the use of Endpoints

# In[19]:


import pandas as pd
from datetime import datetime
import yfinance as yf

#initialize parameters
start_date = datetime(2021,1,4)
end_date = datetime(2021,1,5)

df_data = yf.download('AAPL',start=start_date,end=end_date)

df_data.reset_index(inplace=True)

#creating a dataframe
df_data


# In[20]:


df_data.drop(axis=1, columns=['Adj Close'],inplace=True)
df_data.drop(axis=1, columns=['Date'],inplace=True)

data_features_array = df_data.values
data_features_array


# # Serialize data
# 
# ### Inference -Serialzed Input by Sagemaker Function

# In[22]:


#to get the results
#if we dont use utf-8 then the output will be of bytes format
Y_pred_Fcn = xgb_predictor.predict(data_features_array).decode('utf-8')
Y_pred_Fcn


# ### Inference - Serialized Input by buily-in function (Lambda fucniton friendly) 

# In[28]:


# input is a list of list to allow multiple row of features
Input = [[1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08],
         [1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08],
         [1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08]]


#in lambda funciton we cannot use input serializer so

Serialized_Input = ','.join(map(str,Input[0]))

print(Serialized_Input, type(Serialized_Input))


# In[29]:


Y_pred = xgb_predictor.predict(Serialized_Input).decode('utf-8')
Y_pred


# In[26]:


# so in this session we learned how to create a endpoint ,serilize it and make it lambda friendly


# ## 6.0 Lambda funtion handler
#    1)Trigger Endpoint 
#    
#    2)Trigger SNS

# In[35]:


import boto3

ENDPOINT_NAME = 'sagemaker-xgboost-2024-07-02-09-36-53-593'
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event,context):
    inputs = event['data']  
    result=[]
    for inp in inputs:
        #serializing inputs
        serialized_input = ','.join(map(str,inp))
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                          ContentType='text/csv',
                                          Body=serialized_input)

    #we only need to get the body
        result.append(response['Body'].read().decode())
    return result



Input_json ={'data':
         [[1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08],
         [1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08],
         [1.33520004e+02, 1.33610001e+02, 1.26760002e+02, 1.29410004e+02,1.43301900e+08]]
        }

lambda_handler(Input_json, __)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




