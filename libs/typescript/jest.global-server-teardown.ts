import { rmSync } from 'fs';
import { ChildProcess } from 'child_process';

module.exports = async () => {
  const globals = globalThis as any;
  const mlflowProcess = globals.mlflowProcess as ChildProcess;
  const tempDir = globals.tempDir as string;

  if (mlflowProcess) {
    // Kill the process group to ensure worker processes spawned by uvicorn are terminated
    process.kill(-mlflowProcess.pid!, 'SIGTERM');

    // Wait for 1 second to ensure the process is terminated
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  if (tempDir) {
    rmSync(tempDir, { recursive: true, force: true });
  }
};
