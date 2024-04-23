package server

import (
	"context"
	"errors"
	"sync"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
)

func Launch(ctx context.Context, cfg config.Config) error {
	if len(cfg.PythonCommand) > 0 {
		return launchCommandAndServer(ctx, cfg)
	}
	return launchServer(ctx, cfg)
}

func launchCommandAndServer(ctx context.Context, cfg config.Config) error {
	var errs []error
	var wg sync.WaitGroup

	cmdCtx, cmdCancel := context.WithCancel(ctx)
	srvCtx, srvCancel := context.WithCancel(ctx)

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := launchCommand(cmdCtx, cfg); err != nil && cmdCtx.Err() == nil {
			errs = append(errs, err)
		}
		srvCancel()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := launchServer(srvCtx, cfg); err != nil && srvCtx.Err() == nil {
			errs = append(errs, err)
		}
		cmdCancel()
	}()

	wg.Wait()

	return errors.Join(errs...)
}
