package org.mlflow.spark.autologging

/**
  * Trait defining subscriber interface for receiving information about Spark datasource reads.
  * This trait can be implemented in Python in order to obtain datasource read
  * information, see https://www.py4j.org/advanced_topics.html#implementing-java-interfaces-from-python-callback
  */
trait MlflowAutologEventSubscriber {
  /**
   * Method called on datasource reads.
   * @param path Path of the datasource that was read
   * @param version Version, if applicable (e.g. for Delta tables) of datasource that was read
   * @param format Format ("csv", "json", etc) of the datasource that was read
   */
  def notify(path: String, version: String, format: String): Any

  /**
   * Used to verify that a subscriber is still responsive - for example,
   * in the case of a Python subscriber, invoking the ping() method from Java via a Py4J callback
   * allows us to verify that the associated Python process is still alive.
   */
  def ping(): Unit

  /**
   * Return the ID of the notebook associated with this subscriber, if any. The returned ID is
   * expected to be unique across all subscribers (e.g. a UUID).
   */
  def replId: String
}
