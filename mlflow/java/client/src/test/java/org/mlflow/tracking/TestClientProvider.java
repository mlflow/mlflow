package org.mlflow.tracking;

import java.io.*;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.mlflow.tracking.creds.MlflowHostCredsProvider;

/**
 * Provides an MLflow API client for testing. This is a real client, pointed to a real server.
 * If the MLFLOW_TRACKING_URI environment variable is set, we will talk to the provided server;
 * this allows running tests against existing servers. Otherwise, we will launch a local
 * server on an ephemeral port, and manage its lifecycle.
 */
public class TestClientProvider {
  private static final Logger logger = LoggerFactory.getLogger(TestClientProvider.class);

  private static final long MAX_SERVER_WAIT_TIME_MILLIS = 60 * 1000;

  private Process serverProcess;

  private MlflowClient client;

  /**
   * Intializes an MLflow client and, if necessary, a local MLflow server process as well.
   * Callers should always call {@link #cleanupClientAndServer()}.
   */
  public MlflowClient initializeClientAndServer() throws IOException {
    if (serverProcess != null) {
      throw new IllegalStateException("Previous server process not cleaned up");
    }

    String trackingUri = System.getenv("MLFLOW_TRACKING_URI");
    if (trackingUri != null) {
      logger.info("MLFLOW_TRACKING_URI was set, test will run against that server");
      client = new MlflowClient(trackingUri);
      return client;
    } else {
      Path tempDir = Files.createTempDirectory(getClass().getSimpleName());
      String mlruns = tempDir.resolve("mlruns").toString();
      return startServerProcess(mlruns, mlruns);
    }
  }

  public MlflowClient initializeClientAndSqlLiteBasedServer() throws IOException {
    if (serverProcess != null) {
      throw new IllegalStateException("Previous server process not cleaned up");
    }

    String trackingUri = System.getenv("MLFLOW_TRACKING_URI");
    if (trackingUri != null) {
      logger.info("MLFLOW_TRACKING_URI was set, test will run against that server");
      client = new MlflowClient(trackingUri);
      return client;
    } else {
      Path tempDir = Files.createTempDirectory(getClass().getSimpleName());
      String tempDBFile = tempDir.resolve("sqldb").toAbsolutePath().toString();
      return startServerProcess("sqlite:///" + tempDBFile, tempDir.toString());
    }
  }

  /**
   * Performs any necessary cleanup on the client and server allocated by
   * {@link #initializeClientAndServer()}. This is safe to call even if the client/server were
   * not initialized successfully.
   */
  public void cleanupClientAndServer() throws InterruptedException {
    if (client != null) {
      client.close();
    }
    if (serverProcess == null) {
      return;
    }

    try {
      serverProcess.destroy();
      // Do our best to ensure that the
      boolean processTerminated = serverProcess.waitFor(30, TimeUnit.SECONDS);
      if (!processTerminated) {
        logger.warn("Server process did not terminate in 30 seconds, will forcibly destroy");
        serverProcess.destroyForcibly();
      }
    } catch (Exception ex) {
      logger.warn("Failed to destroy server process nicely.", ex);
      serverProcess.destroyForcibly();
    }
    serverProcess = null;
  }

  public MlflowHostCredsProvider getClientHostCredsProvider(MlflowClient client) {
    return client.getInternalHostCredsProvider();
  }

  /**
   * Launches an "mlflow server" process locally. This requires that the "mlflow" command
   * line client is on the local PATH (e.g., that we're within a conda environment), and that
   * we are allowed to bind to 127.0.0.1 on ephemeral ports.
   *
   * Standard out and error from the server will be streamed to System.out and System.err.
   *
   * This method will wait until the server is up and running
   * @param backendStoreUri the backend store uri to use
   * @param
   * @return MlflowClient pointed at the local server.
   */
  private MlflowClient startServerProcess(String backendStoreUri,
                                          String defaultArtifactRoot) throws IOException {
    ProcessBuilder pb = new ProcessBuilder();
    int freePort = getFreePort();
    String bindAddress = "127.0.0.1";
    pb.command("mlflow", "server",
            "--host", bindAddress,
            "--port", "" + freePort,
            "--backend-store-uri", backendStoreUri,
            "--workers", "1",
            "--default-artifact-root", defaultArtifactRoot);
    serverProcess = pb.start();

    // NB: We cannot use pb.inheritIO() because that interacts poorly with the Maven
    // Surefire test runner (it keeps waiting for more input/output indefinitely).
    // Therefore, we must manually drain the stdout and stderr streams.
    drainStream(serverProcess.getInputStream(), System.out, "mlflow-server-stdout-reader");
    drainStream(serverProcess.getErrorStream(), System.err, "mlflow-server-stderr-reader");

    logger.info("Awaiting start of server on port " + freePort);
    long startTime = System.nanoTime();
    while (System.nanoTime() - startTime < MAX_SERVER_WAIT_TIME_MILLIS * 1000 * 1000) {
      if (isPortOpen(bindAddress, freePort, 1)) {
        break;
      }
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
    if (!isPortOpen(bindAddress, freePort, 1)) {
      serverProcess.destroy();
      throw new IllegalStateException("Server failed to start on port " + freePort + " after "
        + MAX_SERVER_WAIT_TIME_MILLIS + " milliseconds.");
    }

    client = new MlflowClient("http://" + bindAddress + ":" + freePort);
    return client;
  }

  /** Launches a new daemon Thread to drain the given InputStream into the given OutputStream. */
  private void drainStream(InputStream inStream, PrintStream outStream, String threadName) {
    Thread drainThread = new Thread(threadName) {
      @Override
      public void run() {
        BufferedReader reader = new BufferedReader(new InputStreamReader(inStream,
          StandardCharsets.UTF_8));
        reader.lines().forEach(outStream::println);
        logger.info("Drain completed on " + threadName);
      }
    };
    drainThread.setDaemon(true);
    drainThread.start();
  }

  /** Returns an ephemeral port which is very likely to be free, even in cases of contention. */
  private int getFreePort() throws IOException {
    // *nix systems rarely reuse recently allocated ports, so we allocate one and then release it.
    ServerSocket sock = new ServerSocket(0);
    int port = sock.getLocalPort();
    sock.close();
    return port;
  }

  /**
   * Checks if a server is listening on the given host and port. This simply attempts to establish
   * a TCP connection and returns false if it fails to do so within timeoutSeconds.
   */
  private boolean isPortOpen(String host, int port, int timeoutSeconds) {
    try {
      String ip = InetAddress.getByName(host).getHostAddress();
      Socket socket = new Socket();
      socket.connect(new InetSocketAddress(ip, port), timeoutSeconds * 1000);
      socket.close();
    } catch (IOException e) {
      return false;
    }
    return true;
  }
}
