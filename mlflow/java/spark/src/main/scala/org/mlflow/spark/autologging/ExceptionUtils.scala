package org.mlflow.spark.autologging

import java.io.{PrintWriter, StringWriter}

import scala.util.control.NonFatal

import org.slf4j.Logger

private[autologging] object ExceptionUtils {
  /** Helper for generating a nicely-formatted string representation of a Throwable */
  def serializeException(exc: Throwable): String = {
    val sw = new StringWriter
    exc.printStackTrace(new PrintWriter(sw))
    sw.toString
  }

  def getUnexpectedExceptionMessage(exc: Throwable, msg: String): String = {
    s"Unexpected exception $msg. Please report this error, along with the " +
      s"following stacktrace, on https://github.com/mlflow/mlflow/issues:\n" +
      s"${ExceptionUtils.serializeException(exc)}"
  }

  def tryAndLogSilently(logger: Logger, errorMsg: String, fn: => Any): Unit = {
    try {
      fn
    } catch {
      case NonFatal(e) =>
        if (logger.isTraceEnabled) {
          logger.trace(s"Skipping operation $errorMsg: ${e.getMessage}")
        }
    }
  }

  def tryAndLogUnexpectedError(logger: Logger, errorMsg: String, fn: => Any): Unit = {
    try {
      fn
    } catch {
      case NonFatal(e) =>
        if (logger.isTraceEnabled) {
          logger.trace(getUnexpectedExceptionMessage(e, errorMsg))
        }
    }
  }

}
