package main

import (
	"context"
	"encoding/json"
	"os"
	"os/signal"
	"syscall"

	"github.com/sirupsen/logrus"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

func main() {
	var config config.Config

	loggerInstance := logrus.StandardLogger()

	if err := json.Unmarshal([]byte(os.Getenv("MLFLOW_GO_CONFIG")), &config); err != nil {
		loggerInstance.Fatal(err)
	}

	logLevel, err := logrus.ParseLevel(config.LogLevel)
	if err != nil {
		loggerInstance.Fatal(err)
	}

	loggerInstance.SetLevel(logLevel)
	loggerInstance.Warn("The experimental Go server is not yet fully supported and may not work as expected")
	loggerInstance.Debugf("Loaded config: %#v", config)

	ctx, cancel := context.WithCancel(context.Background())
	sigint := make(chan os.Signal, 1)
	signal.Notify(sigint, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigint
		loggerInstance.Info("Shutting down MLflow experimental Go server")
		cancel()
	}()

	loggerInstance.Infof("Starting MLflow experimental Go server on http://%s", config.Address)

	if err := server.Launch(ctx, loggerInstance, &config); err != nil {
		loggerInstance.Fatal(err)
	}
}
