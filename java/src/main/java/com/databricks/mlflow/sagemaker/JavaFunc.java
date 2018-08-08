package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.Flavor;
import com.databricks.mlflow.LoaderModule;
import com.databricks.mlflow.LoaderModuleException;
import com.databricks.mlflow.TrackingUtils;
import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.utils.PackageInstaller;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.List;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Constructor;

class JavaFunc {
    private static final String LOADER_METHOD_NAME = "load";

    protected static Predictor load(String path, Optional<String> runId)
        throws IOException, InvocationTargetException, InstantiationException,
               LoaderModuleException {
        if (runId.isPresent()) {
            // Get the run-relative model logging directory
            try {
                path = TrackingUtils.getModelLogDir(path, runId.get());
            } catch (UnsupportedOperationException e) {
                e.printStackTrace();
                throw new LoaderModuleException(
                    "The model could not be loaded as a java function from a run-relative path");
            }
        }
        Model config = Model.fromRootPath(path);
        Optional<JavaFuncFlavor> javaFuncFlavor =
            config.getFlavor(JavaFuncFlavor.FLAVOR_NAME, JavaFuncFlavor.class);

        if (!javaFuncFlavor.isPresent()) {
            throw new LoaderModuleException(
                String.format("Attempted to load a model using the %s flavor,"
                        + " but  the model does not have this flavor.",
                    JavaFuncFlavor.FLAVOR_NAME));
        }

        return loadModelFromClass(javaFuncFlavor.get().getLoaderClassName(), config);
    }

    private static Predictor loadModelFromClass(String loaderClassName, Model modelConfig)
        throws InvocationTargetException, InstantiationException, LoaderModuleException {
        try {
            Class<?> loaderClass = Class.forName(loaderClassName);
            Constructor<?> cons = loaderClass.getConstructor();
            LoaderModule loaderModule = (LoaderModule) cons.newInstance();
            return loaderModule.load(modelConfig);
        } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException e) {
            e.printStackTrace();
            throw new LoaderModuleException(String.format(
                "Failed to load model using loader module with name %s", loaderClassName));
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
