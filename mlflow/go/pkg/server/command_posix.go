//go:build !windows

package server

import (
	"os/exec"
	"syscall"
)

func setNewProcessGroup(cmd *exec.Cmd) {
	//nolint:exhaustruct
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
		Pgid:    0,
	}
}
