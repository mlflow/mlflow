library(mlflow)

# define parameters
my_int <- mlflow_param("my_int", 1, "integer")
my_num <- mlflow_param("my_num", 1.0, "numeric")
my_str <- mlflow_param("my_str", "a", "string")

# log parameters
mlflow_log_param("param_int", my_int)
mlflow_log_param("param_num", my_num)
mlflow_log_param("param_str", my_str)
