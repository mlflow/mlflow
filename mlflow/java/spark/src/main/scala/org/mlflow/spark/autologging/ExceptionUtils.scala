package org.mlflow.spark.autologging

import java.io.{PrintWriter, StringWriter}

private[autologging] object ExceptionUtils {
  def serializeException(exc: Throwable): String = {
    val sw = new StringWriter
    exc.printStackTrace(new PrintWriter(sw))
    sw.toString
  }
}
