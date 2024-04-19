package server

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
)

func Launch(port int) {
	app := fiber.New()

	var mlflowService MlflowService
	var modelRegistryService ModelRegistryService
	var mlflowArtifactsService MlflowArtifactsService

	registerMlflowServiceRoutes(mlflowService, app)
	registerModelRegistryServiceRoutes(modelRegistryService, app)
	registerMlflowArtifactsServiceRoutes(mlflowArtifactsService, app)

	app.Listen(fmt.Sprintf(":%d", port))
}
