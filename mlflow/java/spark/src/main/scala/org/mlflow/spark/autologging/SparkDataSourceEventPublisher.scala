package org.mlflow.spark.autologging

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent._

import org.apache.spark.SparkContext
import org.apache.spark.scheduler.SparkListener
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory
import py4j.Py4JException

import scala.util.control.NonFatal

/**
  * Object exposing the actual implementation of SparkDatasourceEventPublisher.
  * We opt for this pattern (an object extending a trait) so that we can mock methods of the
  * trait in testing
  */
object SparkDataSourceEventPublisher extends SparkDataSourceEventPublisherImpl {

}

trait SparkDataSourceEventPublisherImpl {
  val logger = LoggerFactory.getLogger(getClass)

  private[autologging] var sparkQueryListener: SparkDataSourceListener = _
  val ex = new ScheduledThreadPoolExecutor(1)
  var subscribers: Map[String, SparkDataSourceEventSubscriber] = Map()
  var scheduledTask: ScheduledFuture[_] = null

  def spark: SparkSession = {
    SparkSession.builder.getOrCreate()
  }

  private[autologging] def getReplIdAwareListener: SparkDataSourceListener = {
    new DatabricksSparkDataSourceListener()
  }

  // Initialize Spark listener that pulls Delta query plan information & bubbles it up to registered
  // Python subscribers
  def init(): Unit = synchronized {
    if (sparkQueryListener == null) {

      // Get SparkContext & determine if REPL id is set - if not, then we log irrespective of repl
      // ID, but if so, we log conditionally on repl ID
      val sc = SparkContext.getOrCreate()
      val replId = Option(sc.getLocalProperty("spark.databricks.replId"))
      val listener = replId match {
        case None => new SparkDataSourceListener()
        case Some(_) => getReplIdAwareListener
      }

      // NB: We take care to set the variable only after adding the Spark listener succeeds,
      // in case listener registration throws
      spark.sparkContext.addSparkListener(listener)
      sparkQueryListener = listener
      // Schedule regular cleanup of detached subscribers, e.g. those associated with detached notebooks
      // TODO: can this throw, and what do we do if/when it does?
      val task = new Runnable {
        def run(): Unit = {
          unregisterBrokenSubscribers()
        }
      }
      scheduledTask = ex.scheduleAtFixedRate(task, 1, 1, TimeUnit.SECONDS)
    }
  }

  def stop(): Unit = synchronized {
    if (sparkQueryListener != null) {
      spark.sparkContext.removeSparkListener(sparkQueryListener)
      sparkQueryListener = null
      while(!scheduledTask.cancel(false)) {
        Thread.sleep(1000)
      }
      subscribers = Map.empty
    }
  }

  def register(subscriber: SparkDataSourceEventSubscriber): String = synchronized {
    if (sparkQueryListener == null) {
      throw new RuntimeException("Please call init() before attempting to register a subscriber")
    }
    this.synchronized {
      val uuid = java.util.UUID.randomUUID().toString
      subscribers = subscribers + (uuid -> subscriber)
      uuid
    }
  }

  /** Unregister subscribers broken e.g. due to detaching of the associated Python REPL */
  def unregisterBrokenSubscribers(): Unit = synchronized {
    val brokenUuids = subscribers.flatMap { case (uuid, listener) =>
      try {
        listener.ping()
        Seq.empty
      } catch {
        case e: Py4JException =>
          logger.info(s"Listener with UUID $uuid not responding to health checks, removing it")
          Seq(uuid)
        case NonFatal(e) =>
          logger.error(s"Unknown exception while checking health of listener $uuid, removing it. " +
            s"Please report this error at https://github.com/mlflow/mlflow/issues, along with " +
            s"the following stacktrace:\n${e.getStackTrace.map(_.toString).mkString("\n")}")
          Seq(uuid)
      }
    }
    brokenUuids.foreach { uuid =>
      subscribers = subscribers - uuid
    }
  }

  def publishEvent(replIdOpt: Option[String], sparkTableInfo: SparkTableInfo): Unit = synchronized {
    sparkTableInfo match {
      case SparkTableInfo(path, version, format) =>
        val time = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss").format(LocalDateTime.now)
        println(s"Notifying ${subscribers.size} listeners about read to ${path}, ${version}, " +
          s"time ${time}. My class: ${getClass.getName}")

        for ((uuid, listener) <- subscribers) {
          try {
            if (replIdOpt.isEmpty || listener.replId == replIdOpt.get) {
              listener.notify(path, version.getOrElse("unknown"), format.getOrElse("unknown"))
            }
          } catch {
            case e: Py4JException =>
              logger.error(s"Unable to forward event to listener with UUID ${uuid}.")
          }
        }
      case _ =>
    }
  }
}