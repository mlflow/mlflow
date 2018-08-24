package com.databricks.mlflow.client.util;

import java.io.*;
import java.util.stream.Collectors;

public class GitVersion {
    public static void main(String [] args) throws Exception {
        System.out.println(version());
    }

    public static String version() throws Exception {
        return (new GitVersion()).getVersion();
    }

    public String getVersion() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        //String file = "git.properties";
        String file = "com.databricks.mlflow.client.git.properties"; // to be able to get uniquely from classpath
        InputStream input = classLoader.getResourceAsStream(file);
        if (input == null) {
            throw new FileNotFoundException("Cannot find resource "+file);
        }
        try (BufferedReader buffer = new BufferedReader(new InputStreamReader(input))) {
            return buffer.lines().collect(Collectors.joining("\n"));
        }
    }
}
