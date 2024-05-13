package server

import (
	"context"
	"errors"
	"fmt"
	"net"
	"path/filepath"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/proxy"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/sirupsen/logrus"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
)

func launchServer(ctx context.Context, cfg config.Config) error {
	app := fiber.New(fiber.Config{
		BodyLimit:             16 * 1024 * 1024,
		ReadBufferSize:        16384,
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          600 * time.Second,
		IdleTimeout:           120 * time.Second,
		ServerHeader:          fmt.Sprintf("mlflow/%s", cfg.Version),
		DisableStartupMessage: true,
		// ErrorHandler: func(c *fiber.Ctx, err error) error {},
	})

	app.Use(compress.New())
	app.Use(recover.New(recover.Config{EnableStackTrace: true}))
	app.Use(logger.New(logger.Config{
		Format: "${status} - ${latency} ${method} ${path}\n",
		Output: logrus.StandardLogger().Writer(),
	}))

	mlflowService, err := NewMlflowService(cfg.StoreUrl)
	if err != nil {
		return err
	}

	contract.RegisterMlflowServiceRoutes(mlflowService, app)
	contract.RegisterModelRegistryServiceRoutes(modelRegistryService, app)
	contract.RegisterMlflowArtifactsServiceRoutes(mlflowArtifactsService, app)

	app.Static("/static-files", cfg.StaticFolder)
	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendFile(filepath.Join(cfg.StaticFolder, "index.html"))
	})

	app.Use(proxy.BalancerForward([]string{cfg.PythonAddress}))

	go func() {
		<-ctx.Done()
		if err := app.ShutdownWithTimeout(cfg.ShutdownTimeout.Duration); err != nil {
			logrus.Errorf("Failed to gracefully shutdown MLflow experimental Go server: %v", err)
		}
	}()

	// Wait until the Python server is ready
	for {
		dialer := &net.Dialer{}
		conn, err := dialer.DialContext(ctx, "tcp", cfg.PythonAddress)
		if err == nil {
			conn.Close()
			break
		}
		if errors.Is(err, context.Canceled) {
			return err
		}
		time.Sleep(1 * time.Second)
	}
	logrus.Debugf("Python server is ready")

	return app.Listen(cfg.Address)
}
