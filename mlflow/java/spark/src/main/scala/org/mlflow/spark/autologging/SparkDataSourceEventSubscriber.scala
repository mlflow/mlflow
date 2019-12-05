package org.mlflow.spark.autologging

/* You can add arbitrary methods here,
 * as long as these match corresponding Python interface
 */
trait SparkDataSourceEventSubscriber {
  /* This will be implemented by a Python class.
   * You can of course use more specific types,
   * for example here String => Unit */
  def notify(path: String, version: String, format: String): Any

  def ping(): Unit

  def replId: String
}