import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/rohinibhosale1223/miniproject.mlflow")
dagshub.init(repo_owner='rohinibhosale1223', repo_name='miniproject', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
