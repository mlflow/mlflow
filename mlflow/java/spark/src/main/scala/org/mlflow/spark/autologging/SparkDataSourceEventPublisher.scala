package org.mlflow.spark.autologging

import java.util.concurrent._

import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import py4j.Py4JException

import scala.collection.mutable
import scala.util.control.NonFatal

/**
  * Object exposing the actual implementation of SparkDatasourceEventPublisher.
  * We opt for this pattern (an object extending a trait) so that we can mock methods of the
  * trait in testing
  */
object SparkDataSourceEventPublisher extends SparkDataSourceEventPublisherImpl {

}

/**
 * Trait implementing a publisher interface for publishing events on Spark datasource reads to
 * a set of listeners. See the design doc:
 * https://docs.google.com/document/d/11nhwZtj-rps0stxuIioFBM9lkvIh_ua45cAFy_PqdHU/edit for more
 * details.
 */
private[autologging] trait SparkDataSourceEventPublisherImpl {
  private val logger = LoggerFactory.getLogger(getClass)

  private[autologging] var sparkQueryListener: SparkDataSourceListener = _
  private val ex = new ScheduledThreadPoolExecutor(1)
  private[autologging] var subscribers: mutable.LinkedHashMap[String, SparkDataSourceEventSubscriber] =
    mutable.LinkedHashMap[String, SparkDataSourceEventSubscriber]()
  private var scheduledTask: ScheduledFuture[_] = _

  def spark: SparkSession = {
    SparkSession.builder.getOrCreate()
  }

  // Exposed for testing
  private[autologging] def getSparkDataSourceListener: SparkDataSourceListener = {
    new SparkDataSourceListener()
  }

  // Initialize Spark listener that pulls Delta query plan information & bubbles it up to registered
  // Python subscribers, along with a GC loop for removing unrespoins
  def init(gcDeadSubscribersIntervalSec: Int = 1): Unit = synchronized {
    if (sparkQueryListener == null) {
      val listener = getSparkDataSourceListener
      // NB: We take care to set the variable only after adding the Spark listener succeeds,
      // in case listener registration throws. This is defensive - adding a listener should
      // always succeed.
      spark.sparkContext.addSparkListener(listener)
      sparkQueryListener = listener
      // Schedule regular cleanup of detached subscribers, e.g. those associated with detached
      // notebooks
      val task = new Runnable {
        def run(): Unit = {
          unregisterBrokenSubscribers()
        }
      }
      scheduledTask = ex.scheduleAtFixedRate(task, 1, gcDeadSubscribersIntervalSec, TimeUnit.SECONDS)
    }
  }

  def stop(): Unit = synchronized {
    if (sparkQueryListener != null) {
      spark.sparkContext.removeSparkListener(sparkQueryListener)
      sparkQueryListener = null
      while(!scheduledTask.cancel(false)) {
        Thread.sleep(1000)
      }
      subscribers = mutable.LinkedHashMap()
    }
  }

  def register(subscriber: SparkDataSourceEventSubscriber): String = synchronized {
    if (sparkQueryListener == null) {
      throw new RuntimeException("Please call init() before attempting to register a subscriber")
    }
    this.synchronized {
      val uuid = java.util.UUID.randomUUID().toString
      subscribers.put(uuid, subscriber)
      uuid
    }
  }

  /** Unregister subscribers broken e.g. due to detaching of the associated Python REPL */
  private[autologging] def unregisterBrokenSubscribers(): Unit = synchronized {
    val brokenUuids = subscribers.flatMap { case (uuid, listener) =>
      try {
        listener.ping()
        Seq.empty
      } catch {
        case e: Py4JException =>
          logger.info(s"Subscriber with UUID $uuid not responding to health checks, removing it")
          Seq(uuid)
        case NonFatal(e) =>
          logger.error(s"Unexpected exception while checking health of subscriber $uuid, " +
            s"removing it. Please report this error at https://github.com/mlflow/mlflow/issues, " +
            s"along with the following stacktrace:\n" +
            s"${ExceptionUtils.serializeException(e)}")
          Seq(uuid)
      }
    }
    brokenUuids.foreach { uuid =>
      subscribers.remove(uuid)
    }
  }

  private[autologging] def publishEvent(
      replIdOpt: Option[String],
      sparkTableInfo: SparkTableInfo): Unit = synchronized {
    sparkTableInfo match {
      case SparkTableInfo(path, version, format) =>
        for ((uuid, listener) <- subscribers) {
          try {
            if (replIdOpt.isEmpty || listener.replId == replIdOpt.get) {
              listener.notify(path, version.getOrElse("unknown"), format.getOrElse("unknown"))
            }
          } catch {
            case e: Py4JException =>
              logger.error(s"Unable to forward event to listener with UUID $uuid.")
          }
        }
      case _ =>
    }
  }
}
