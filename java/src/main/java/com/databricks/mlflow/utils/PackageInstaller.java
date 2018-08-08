package com.databricks.mlflow.utils;

import java.util.Arrays;
import java.util.List;

import org.apache.maven.cli.MavenCli;

public class PackageInstaller {
    public static void installPackage(String mavenCoordinate) {
        installPackages(Arrays.asList(mavenCoordinate));
    }

    public static void installPackages(List<String> mavenCoordinates) {
        MavenCli cli = new MavenCli();
        for (String coordinate : mavenCoordinates) {
            installPackage(coordinate, cli);
        }
    }

    private static void installPackage(String coordinate, MavenCli cli) {
        String workingDir = System.getProperty("user.dir");
        System.out.println(coordinate);
        System.out.println(
            cli.doMain(new String[] {"install", coordinate}, workingDir, System.out, System.out));
    }
}
