package main

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

func main() {
	configuration := server.LaunchConfiguration{
		Port:         5001,
		PythonPort:   4001,
		StaticFolder: "/workspaces/mlflow/mlflow/server/js/build",
	}
	server.Launch(configuration)
}
