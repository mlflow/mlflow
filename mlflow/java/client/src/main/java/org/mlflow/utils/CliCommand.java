package org.mlflow.utils;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import org.apache.commons.io.IOUtils;
import org.mlflow.tracking.MlflowClientException;
import org.mlflow.tracking.creds.MlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCredsProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs MLflow CLI commands
 *
 * TODO(andy): CliBasedArtifactRepository was the progenitor to this function. It contains
 *              numerous similar functions but the class is tied to runId so it is not
 *              usable for things such as downloading artifacts using only the artifact-uri.
 *
 */
public class CliCommand implements Callable<String> {
    // Global check if we ever successfully loaded 'mlflow'. This allows us to print a more
    // helpful error message if the executable is not in the path.
    private static final AtomicBoolean mlflowSuccessfullyLoaded = new AtomicBoolean(false);

    private static final Logger logger = LoggerFactory.getLogger(CliCommand.class);

    private final String PYTHON_EXECUTABLE =
            Optional.ofNullable(System.getenv("MLFLOW_PYTHON_EXECUTABLE")).orElse("python");

    private final String PYTHON_COMMAND = "mlflow.cli";

    private final List<String> args;

    private final MlflowHostCredsProvider hostCredsProvider;

    public CliCommand(List<String> args, MlflowHostCredsProvider hostCredsProvider) {
        this.args = args;
        this.hostCredsProvider = hostCredsProvider;
    }

    @Override
    public String call() {
        checkMlflowAccessible();

        String stdout;
        Process process = null;

        try {
            ProcessBuilder pb = createMlflowCliCommand(args);
            process = pb.start();
            stdout = IOUtils.toString(process.getInputStream(), StandardCharsets.UTF_8);
            int exitValue = process.waitFor();
            if (exitValue != 0) {
                throw new MlflowClientException("Failed to run mlflow cli command. Error: " +
                        getErrorBestEffort(process));
            }
        } catch (IOException | InterruptedException e) {
            throw new MlflowClientException("Failed to fork mlflow process" +
                    ". Process stderr: " + getErrorBestEffort(process), e);
        }
        return stdout;
    }

    /** Does our best to get the process's stderr, or returns a dummy return value. */
    private String getErrorBestEffort(Process process) {
        if (process == null) {
            return "<process not started>";
        }
        try {
            return IOUtils.toString(process.getErrorStream(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return "<error unknown>";
        }
    }

    /**
     * Checks whether the 'mlflow' executable is available, and throws a nice error if not.
     * If this method has ever run successfully before (in the entire JVM), we will not rerun it.
     */
    private void checkMlflowAccessible(){
        if (mlflowSuccessfullyLoaded.get()) {
            return;
        }

        try {
            ProcessBuilder pb = createMlflowCliCommand(Lists.newArrayList("--help"));
            pb.start();
            logger.info("Found local mlflow executable");
            mlflowSuccessfullyLoaded.set(true);
        } catch (IOException | MlflowClientException e) {
            String errorMessage = String.format("Failed to exec '%s -m %s', Please make" +
                            " sure mlflow is available on your local system path " +
                            "(e.g., from 'pip install mlflow')",
                    PYTHON_EXECUTABLE, PYTHON_COMMAND);
            throw new MlflowClientException(errorMessage, e);
        }
    }

    @VisibleForTesting
    void setProcessEnvironment(Map<String, String> environment) {
        MlflowHostCreds hostCreds = hostCredsProvider.getHostCreds();
        environment.put("MLFLOW_TRACKING_URI", hostCreds.getHost());
        if (hostCreds.getUsername() != null) {
            environment.put("MLFLOW_TRACKING_USERNAME", hostCreds.getUsername());
        }
        if (hostCreds.getPassword() != null) {
            environment.put("MLFLOW_TRACKING_PASSWORD", hostCreds.getPassword());
        }
        if (hostCreds.getToken() != null) {
            environment.put("MLFLOW_TRACKING_TOKEN", hostCreds.getToken());
        }
        if (hostCreds.shouldIgnoreTlsVerification()) {
            environment.put("MLFLOW_TRACKING_INSECURE_TLS", "true");
        }
    }

    ProcessBuilder createMlflowCliCommand(List<String> args) {
        List<String> cmds = Lists.newArrayList(PYTHON_EXECUTABLE, "-m", PYTHON_COMMAND);
        cmds.addAll(args);
        ProcessBuilder processBuilder = new ProcessBuilder(cmds);
        setProcessEnvironment(processBuilder.environment());
        return processBuilder;
    }
}
