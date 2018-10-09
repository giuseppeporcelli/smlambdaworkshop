# Training the Spam Filter model with Amazon SageMaker

## Overview

Amazon SageMaker is a fully-managed service that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale. Amazon SageMaker removes all the barriers that typically slow down developers who want to use machine learning.

Machine learning often feels a lot harder than it should be to most developers because the process to build and train models, and then deploy them into production is too complicated and too slow. First, you need to collect and prepare your training data to discover which elements of your data set are important. Then, you need to select which algorithm and framework you’ll use. After deciding on your approach, you need to teach the model how to make predictions by training, which requires a lot of compute. Then, you need to tune the model so it delivers the best possible predictions, which is often a tedious and manual effort. After you’ve developed a fully trained model, you need to integrate the model with your application and deploy this application on infrastructure that will scale. All of this takes a lot of specialized expertise, access to large amounts of compute and storage, and a lot of time to experiment and optimize every part of the process. In the end, it's not a surprise that the whole thing feels out of reach for most developers.

Amazon SageMaker removes the complexity that holds back developer success with each of these steps. Amazon SageMaker includes modules that can be used together or independently to build, train, and deploy your machine learning models.

In this section, we will walk you through creating and training a spam filter machine learning model with Amazon SageMaker.

## The Dataset
In order to train the model, we will use the **UCI SMS Spam** dataset.

Add details.


## Building the model

### Upload the DataSet to Amazon S3
In this section, we will upload the SMS Spam dataset into an Amazon S3 bucket. Amazon SageMaker uses **Amazon S3** as the main storage for both data and model artifacts; you can actually use other sources when loading data into the Jupyter notebook instances, but this is outside of the scope of this lab. 

1.	If you have not downloaded the SMS Spam dataset yet, be sure to download it from here. 
2.	Sign into the **AWS Management Console** and open the **Amazon S3** console at https://console.aws.amazon.com/s3
3.	In the upper-right corner of the AWS Management Console, confirm you are in the desired AWS region (e.g., Ireland).
4.	Now, we will need to create a bucket. In the S3 console, click the **Create Bucket** button. 
5.	For the **Bucket Name**, type sagemaker-lab-<your-initials> in the text box and click Next (take note of the bucket name, it will be needed later for loading data in the notebook instance). Leave everything default in the next 2 pages and click **Create Bucket** in Review page.
6.	Click the link for the bucket name you just created, then click **Upload**.  
7.	Click **Add Files**, find and select the *wind\_turbine\_training\_data\_v2.csv* (please note that you can find this file in the .zip compressed archive you downloaded at step 1) file and click **Upload**.
8.	Wait until the file is uploaded.

### Create a managed Jupyter Notebook instance

Add details.

### Download notebook and run training

Add details.