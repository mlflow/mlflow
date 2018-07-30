package com.databricks.mlflow.javafunc;

import com.databricks.mlflow.Flavor;
import com.databricks.mlflow.LoaderModuleException;
import com.databricks.mlflow.TrackingUtils;
import com.databricks.mlflow.models.Predictor;
import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.utils.PackageInstaller;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.List;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class JavaFunc {
    private static final String LOADER_METHOD_NAME = "load";

    public static Predictor load(String path, Optional<String> runId)
        throws IOException, InvocationTargetException, LoaderModuleException {
        if (runId.isPresent()) {
            // Get the run-relative model logging directory
            path = TrackingUtils.getModelLogDir(path, runId.get());
        }
        String configPath = path + File.separator + "MLmodel";
        Model config = Model.fromPath(configPath);
        Optional<JavaFuncFlavor> javaFuncFlavor =
            config.getFlavor(JavaFuncFlavor.FLAVOR_NAME, JavaFuncFlavor.class);

        if (!javaFuncFlavor.isPresent()) {
            // throw new Exception();
        }

        return loadModelFromClass(javaFuncFlavor.get().getLoaderClassName(), config);
    }

    private static Predictor loadModelFromClass(String loaderClassName, Model modelConfig)
        throws InvocationTargetException, LoaderModuleException {
        try {
            Class<?> loaderClass = Class.forName(loaderClassName);
            Method loaderMethod = loaderClass.getMethod(LOADER_METHOD_NAME, Model.class);
            Predictor predictor = (Predictor) loaderMethod.invoke(null, modelConfig);
            return predictor;
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
