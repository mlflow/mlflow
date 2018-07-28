package com.databricks.mlflow.javafunc;

import com.databricks.mlflow.Flavor;
import com.databricks.mlflow.TrackingUtils;
import com.databricks.mlflow.models.JavaModel;
import com.databricks.mlflow.models.ModelConfig;
import com.databricks.mlflow.utils.PackageInstaller;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.List;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class JavaFunc {
    private static final String LOADER_METHOD_NAME = "load";

    public static JavaModel load(String path, Optional<String> runId)
        throws IOException, InvocationTargetException, LoaderModuleException {
        if (runId.isPresent()) {
            // Get the run-relative model logging directory
            path = TrackingUtils.getModelLogDir(path, runId.get());
        }
        String configPath = path + File.separator + "MLmodel";
        ModelConfig config = ModelConfig.fromPath(configPath);
        Optional<JavaFuncConfig> javaFuncFlavor =
            config.getFlavor(JavaFuncConfig.FLAVOR_NAME, JavaFuncConfig.class);

        if (!javaFuncFlavor.isPresent()) {
            // throw new Exception();
        }

        installPackageDependencies(javaFuncFlavor.get().getPackageDependencies());
        // "IMPORT" CODE SPECIFIED
        // BY javaFuncFlavor.getCodePath()

        return loadModelFromClass(
            javaFuncFlavor.get().getLoaderClassName(), javaFuncFlavor.get().getModelDataPath());
    }

    private static void installPackageDependencies(List<String> packageDependencies) {
        PackageInstaller.installPackages(packageDependencies);
    }

    private static JavaModel loadModelFromClass(String loaderClassName, String modelPath)
        throws InvocationTargetException, LoaderModuleException {
        try {
            Class<?> loaderClass = Class.forName(loaderClassName);
            Method loaderMethod = loaderClass.getMethod(LOADER_METHOD_NAME, String.class);
            JavaModel model = (JavaModel) loaderMethod.invoke(null, modelPath);
            return model;
        } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException e) {
            e.printStackTrace();
            throw new LoaderModuleException(loaderClassName);
        }
    }

    public static void main(String[] args) {
        try {
            String path = args[0];
            JavaFunc.load(path, Optional.<String>empty());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
