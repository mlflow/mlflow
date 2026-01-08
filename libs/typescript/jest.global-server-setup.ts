import { spawn } from 'child_process';
import { tmpdir } from 'os';
import { mkdtempSync } from 'fs';
import { join } from 'path';
import { TEST_PORT, TEST_TRACKING_URI } from './core/tests/helper';

/**
 * Start MLflow Python server. This is necessary for testing Typescript SDK because
 * the SDK does not have a server implementation and talks to the Python server instead.
 */
module.exports = async () => {
  const tempDir = mkdtempSync(join(tmpdir(), 'mlflow-test-'));

  const mlflowRoot = join(__dirname, '../..'); // Use the local dev version

  // Only start a server if one is not already running
  try {
    const response = await fetch(TEST_TRACKING_URI);
    if (response.ok) {
      return;
    }
  } catch (error) {
    // Ignore error
  }

  // eslint-disable-next-line no-console
  console.log(`Starting MLflow server on port ${TEST_PORT}. This may take a few seconds...
      To speed up the test, you can manually start the server and keep it running during local development.`);

  const mlflowProcess = spawn(
    'uv',
    ['run', '--directory', mlflowRoot, 'mlflow', 'server', '--port', TEST_PORT.toString()],
    {
      cwd: tempDir,
      stdio: 'inherit',
      // Create a new process group so we can kill the entire group
      detached: true
    }
  );

  try {
    await waitForServer(TEST_PORT);
    // eslint-disable-next-line no-console
    console.log(`MLflow server is ready on port ${TEST_PORT}`);
  } catch (error) {
    console.error('Failed to start MLflow server:', error);
    throw error;
  }

  // Set global variables for cleanup in jest.global-teardown.ts
  const globals = globalThis as any;
  globals.mlflowProcess = mlflowProcess;
  globals.tempDir = tempDir;
};

async function waitForServer(maxAttempts: number = 30): Promise<void> {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch(TEST_TRACKING_URI);
      if (response.ok) {
        return;
      }
    } catch (error) {
      // Ignore error
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error('Failed to start MLflow server');
}
