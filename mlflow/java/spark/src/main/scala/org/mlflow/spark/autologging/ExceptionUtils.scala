package org.mlflow.spark.autologging

import java.io.{PrintWriter, StringWriter}

private[autologging] object ExceptionUtils {
  /** Helper for generating a nicely-formatted string representation of a Throwable */
  def serializeException(exc: Throwable): String = {
    val sw = new StringWriter
    exc.printStackTrace(new PrintWriter(sw))
    sw.toString
  }
}
