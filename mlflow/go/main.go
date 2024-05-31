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
	if err := json.Unmarshal([]byte(os.Getenv("MLFLOW_GO_CONFIG")), &config); err != nil {
		logrus.Fatal(err)
	}

	logLevel, err := logrus.ParseLevel(config.LogLevel)
	if err != nil {
		logrus.Fatal(err)
	}

	logrus.SetLevel(logLevel)
	logrus.Warn("The experimental Go server is not yet fully supported and may not work as expected")
	logrus.Debugf("Loaded config: %#v", config)

	ctx, cancel := context.WithCancel(context.Background())
	sigint := make(chan os.Signal, 1)
	signal.Notify(sigint, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigint
		logrus.Info("Shutting down MLflow experimental Go server")
		cancel()
	}()

	logrus.Infof("Starting MLflow experimental Go server on http://%s", config.Address)

	if err := server.Launch(ctx, &config); err != nil {
		logrus.Fatal(err)
	}
}
