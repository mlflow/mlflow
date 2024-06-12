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
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/service"
)

func configureApp(loggerInstance *logrus.Logger, cfg *config.Config) (*fiber.App, error) {
	//nolint:mnd
	app := fiber.New(fiber.Config{
		BodyLimit:             16 * 1024 * 1024,
		ReadBufferSize:        16384,
		ReadTimeout:           5 * time.Second,
		WriteTimeout:          600 * time.Second,
		IdleTimeout:           120 * time.Second,
		ServerHeader:          "mlflow/" + cfg.Version,
		DisableStartupMessage: true,
	})

	app.Use(compress.New())
	app.Use(recover.New(recover.Config{EnableStackTrace: true}))
	app.Use(logger.New(logger.Config{
		Format: "${status} - ${latency} ${method} ${path}\n",
		Output: loggerInstance.Writer(),
	}))

	apiApp, err := newAPIApp(loggerInstance, cfg)
	if err != nil {
		return nil, err
	}

	app.Mount("/api/2.0", apiApp)
	app.Mount("/ajax-api/2.0", apiApp)

	app.Static("/static-files", cfg.StaticFolder)
	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendFile(filepath.Join(cfg.StaticFolder, "index.html"))
	})

	app.Get("/health", func(c *fiber.Ctx) error {
		return c.SendString("OK")
	})
	app.Get("/version", func(c *fiber.Ctx) error {
		return c.SendString(cfg.Version)
	})

	app.Use(proxy.BalancerForward([]string{cfg.PythonAddress}))

	return app, nil
}

func launchServer(ctx context.Context, logger *logrus.Logger, cfg *config.Config) error {
	app, err := configureApp(logger, cfg)
	if err != nil {
		return err
	}

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
			return fmt.Errorf("could not connect to Python server: %w", err)
		}

		time.Sleep(1 * time.Second)
	}
	logrus.Debugf("Python server is ready")

	err = app.Listen(cfg.Address)
	if err != nil {
		return fmt.Errorf("failed to start MLflow experimental Go server: %w", err)
	}

	return nil
}

func newFiberConfig() fiber.Config {
	return fiber.Config{
		ErrorHandler: func(context *fiber.Ctx, err error) error {
			var contractError *contract.Error
			if !errors.As(err, &contractError) {
				code := protos.ErrorCode_INTERNAL_ERROR

				var f *fiber.Error
				if errors.As(err, &f) {
					switch f.Code {
					case fiber.StatusBadRequest:
						code = protos.ErrorCode_BAD_REQUEST
					case fiber.StatusServiceUnavailable:
						code = protos.ErrorCode_SERVICE_UNDER_MAINTENANCE
					case fiber.StatusNotFound:
						code = protos.ErrorCode_ENDPOINT_NOT_FOUND
					}
				}

				contractError = contract.NewError(code, err.Error())
			}

			var logFn func(format string, args ...any)

			switch contractError.StatusCode() {
			case fiber.StatusBadRequest:
				logFn = logrus.Infof
			case fiber.StatusServiceUnavailable:
				logFn = logrus.Warnf
			case fiber.StatusNotFound:
				logFn = logrus.Debugf
			default:
				logFn = logrus.Errorf
			}

			logFn("Error encountered in %s %s: %s", context.Method(), context.Path(), err)

			return context.Status(contractError.StatusCode()).JSON(contractError)
		},
	}
}

func newAPIApp(logger *logrus.Logger, cfg *config.Config) (*fiber.App, error) {
	app := fiber.New(newFiberConfig())

	parser, err := NewHTTPRequestParser()
	if err != nil {
		return nil, err
	}

	mlflowService, err := service.NewTrackingService(logger, cfg)
	if err != nil {
		return nil, err
	}

	contract.RegisterMlflowServiceRoutes(mlflowService, parser, app)
	contract.RegisterModelRegistryServiceRoutes(service.ModelRegistryService, parser, app)
	contract.RegisterMlflowArtifactsServiceRoutes(service.MlflowArtifactsService, parser, app)

	return app, nil
}
