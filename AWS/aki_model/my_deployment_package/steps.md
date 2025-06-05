1. tar -czvf model.tar.gz inference.py model_artifact/model.xgb

2. aws s3 cp model.tar.gz s3://<your-bucket-name>/<your-prefix>/model.tar.gz

3. aws sagemaker create-model \
   --model-name aki-xgb‐model \
   --primary-container Image=683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost-inference:1.6-1-cpu-py3,ModelDataUrl=s3://<your-bucket-name>/<your-prefix>/model.tar.gz,Environment="{SAGEMAKER_PROGRAM=inference.py,SAGEMAKER_REGION=<region>}" \
   --execution-role-arn arn:aws:iam::<YOUR_AWS_ACCOUNT_ID>:role/SageMakerExecutionRole
