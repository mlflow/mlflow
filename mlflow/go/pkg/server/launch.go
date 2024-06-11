package server

import (
	"context"
	"errors"
	"sync"

	"github.com/sirupsen/logrus"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
)

func Launch(ctx context.Context, logger *logrus.Logger, cfg *config.Config) error {
	if len(cfg.PythonCommand) > 0 {
		return launchCommandAndServer(ctx, logger, cfg)
	}

	return launchServer(ctx, logger, cfg)
}

func launchCommandAndServer(ctx context.Context, logger *logrus.Logger, cfg *config.Config) error {
	var errs []error

	var waitGroup sync.WaitGroup

	cmdCtx, cmdCancel := context.WithCancel(ctx)
	srvCtx, srvCancel := context.WithCancel(ctx)

	waitGroup.Add(1)

	go func() {
		defer waitGroup.Done()

		if err := launchCommand(cmdCtx, cfg); err != nil && cmdCtx.Err() == nil {
			errs = append(errs, err)
		}

		srvCancel()
	}()

	waitGroup.Add(1)

	go func() {
		defer waitGroup.Done()

		if err := launchServer(srvCtx, logger, cfg); err != nil && srvCtx.Err() == nil {
			errs = append(errs, err)
		}

		cmdCancel()
	}()

	waitGroup.Wait()

	return errors.Join(errs...)
}
