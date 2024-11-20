import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/usmanuuu52/mlops-project.mlflow")
dagshub.init(repo_owner="usmanuuu52", repo_name="mlops-project", mlflow=True)


with mlflow.start_run():
    mlflow.log_param("parameter name", "value")
    mlflow.log_metric("metric name", 1)
