# ML_sagemaker_stockprice
I have used machine learning with the XGBoost algorithm to predict stock prices. Here's a look into our architecture:

![Architecture](https://github.com/ansel9618/ML_sagemaker_stockprice/blob/main/architecture.png)

1. **User Interaction**: Clients send REST-style requests to an API Gateway endpoint-Postman.
2. **API Gateway**: The gateway triggers a Lambda function.
3. **Lambda Function**: The function formats the request for the Amazon SageMaker endpoint, which performs the stock price prediction.
4. **SageMaker Endpoint**: Receives the request and returns the prediction.
5. **Response Handling**: The Lambda function processes the prediction and responds back to the client with a JSON response via API Gateway and sends an email notification through SNS.

# Notification
![](https://github.com/ansel9618/ML_Sagemaker_Stockprice/blob/main/email_notiofication.png)

This integration ensures real-time, accurate predictions and instant notifications for users. The core objective is to create an ML-powered REST API using Amazon API Gateway and Amazon SageMaker,lambda,SNS enabling day-ahead stock price predictions.
