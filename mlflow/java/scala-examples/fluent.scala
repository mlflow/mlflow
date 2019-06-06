//
// Download dependencies (mvn dependency:copy-dependencies) and package java library (mvn package -DskipTests)
// Run with:
// MLFLOW_TRACKING_URI="http://localhost:5000" scala  -cp "client/target/dependency/*:client/target/mlflow-client-1.0.0.jar" scala-examples/fluent.scala
//

import org.mlflow.tracking.MlflowTrackingContext

val mlflow = new MlflowTrackingContext()
mlflow.withActiveRun("a", activeRun => {
  activeRun.logParam("scala", "works")
})