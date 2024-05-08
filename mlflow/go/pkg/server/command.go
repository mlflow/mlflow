package server

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
)

func launchCommand(ctx context.Context, cfg config.Config) error {
	cmd := exec.CommandContext(ctx, cfg.PythonCommand[0], cfg.PythonCommand[1:]...)
	cmd.Env = append(os.Environ(), cfg.PythonEnv...)
	cmd.Stdout = logrus.StandardLogger().Writer()
	cmd.Stderr = logrus.StandardLogger().Writer()
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		logrus.Debug("Sending termination signal to command")
		switch runtime.GOOS {
		case "windows":
			return cmd.Process.Kill()
		default:
			return cmd.Process.Signal(syscall.SIGTERM)
		}
	}
	setNewProcessGroup(cmd)

	logrus.Debugf("Launching command: %v", cmd)
	if err := cmd.Start(); err != nil {
		return err
	}

	return cmd.Wait()
}
