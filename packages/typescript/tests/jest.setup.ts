import { spawn, ChildProcess } from 'child_process';
import { tmpdir } from 'os';
import { mkdtempSync, rmSync } from 'fs';
import { join } from 'path';

let mlflowProcess: ChildProcess;
let tempDir: string;

const TEST_PORT = 5000;
export const TEST_TRACKING_URI = `http://localhost:${TEST_PORT}`;

/**
 * Start MLflow Python server. This is necessary for testing Typescript SDK because
 * the SDK does not have a server implementation and talks to the Python server instead.
 */
beforeAll(async () => {
  tempDir = mkdtempSync(join(tmpdir(), 'mlflow-test-'));
  const mlflowRoot = join(__dirname, '../../..'); // Use the local dev version

  // Only start a server if one is not already running
  // check if tracking uri
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
  mlflowProcess = spawn(
    'uv',
    [
      'run',
      '--with',
      mlflowRoot,
      '--python',
      '3.10',
      'mlflow',
      'server',
      '--port',
      TEST_PORT.toString(),
      '--backend-store-uri',
      `sqlite:///${join(tempDir, 'mlflow.db')}`
    ],
    {
      cwd: tempDir,
      stdio: 'inherit'
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
}, 30000);

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

afterAll(() => {
  if (mlflowProcess) {
    mlflowProcess.kill();
  }
  if (tempDir) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});
