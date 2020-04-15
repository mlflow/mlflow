package org.mlflow.spark.autologging

import java.util.concurrent.{ConcurrentHashMap, ScheduledFuture, ScheduledThreadPoolExecutor, TimeUnit}

import py4j.Py4JException
import org.apache.spark.scheduler.SparkListener

import scala.collection.JavaConverters._
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.util.control.NonFatal

/**
  * Object exposing the actual implementation of MlflowAutologEventPublisher.
  * We opt for this pattern (an object extending a trait) so that we can mock methods of the
  * trait in testing
  */
object MlflowAutologEventPublisher extends MlflowAutologEventPublisherImpl {

}

/**
 * Trait implementing a publisher interface for publishing events on Spark datasource reads to
 * a set of listeners. See the design doc:
 * https://docs.google.com/document/d/11nhwZtj-rps0stxuIioFBM9lkvIh_ua45cAFy_PqdHU/edit for more
 * details.
 */
private[autologging] trait MlflowAutologEventPublisherImpl {
  private val logger = LoggerFactory.getLogger(getClass)

  private[autologging] var sparkQueryListener: SparkListener = _
  private val executor = new ScheduledThreadPoolExecutor(1)
  private[autologging] val subscribers =
    new ConcurrentHashMap[String, MlflowAutologEventSubscriber]()
  private var scheduledTask: ScheduledFuture[_] = _

  def spark: SparkSession = {
    SparkSession.getActiveSession.getOrElse(throw new RuntimeException("Unable to get active " +
      "SparkSession. Please ensure you've started a SparkSession via " +
      "SparkSession.builder.getOrCreate() before attempting to initialize Spark datasource " +
      "autologging."))
  }

  // Exposed for testing
  private[autologging] def getSparkDataSourceListener: SparkListener = {
    // Get SparkContext & determine if REPL id is set - if not, then we log irrespective of repl
    // ID, but if so, we log conditionally on repl ID
    val sc = spark.sparkContext
    val replId = Option(sc.getLocalProperty("spark.databricks.replId"))
    replId match {
      case None => new SparkDataSourceListener(this)
      case Some(_) => new ReplAwareSparkDataSourceListener(this)
    }
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
      scheduledTask = executor.scheduleAtFixedRate(
        task, gcDeadSubscribersIntervalSec, gcDeadSubscribersIntervalSec, TimeUnit.SECONDS)
    }
  }

  def stop(): Unit = synchronized {
    if (sparkQueryListener != null) {
      spark.sparkContext.removeSparkListener(sparkQueryListener)
      sparkQueryListener = null
      while(!scheduledTask.cancel(false)) {
        Thread.sleep(1000)
        logger.info("Unable to cancel task for GC of unresponsive subscribers, retrying...")
      }
      subscribers.clear()
    }
  }

  def register(subscriber: MlflowAutologEventSubscriber): Unit = synchronized {
    if (sparkQueryListener == null) {
      throw new RuntimeException("Please call init() before attempting to register a subscriber")
    }
    subscribers.put(subscriber.replId, subscriber)
  }

  // Exposed for testing - in particular, so that we can iterate over subscribers in a specific
  // order within tests
  private[autologging] def getSubscribers: Seq[(String, MlflowAutologEventSubscriber)] = {
    subscribers.asScala.toSeq
  }

  /** Unregister subscribers broken e.g. due to detaching of the associated Python REPL */
  private[autologging] def unregisterBrokenSubscribers(): Unit = {
    val brokenReplIds = getSubscribers.flatMap { case (replId, listener) =>
      try {
        listener.ping()
        Seq.empty
      } catch {
        case e: Py4JException =>
          logger.info(s"Subscriber with repl ID $replId not responding to health checks, " +
            s"removing it")
          Seq(replId)
        case NonFatal(e) =>
          val msg = ExceptionUtils.getUnexpectedExceptionMessage(e, "while checking health " +
            s"of subscriber with repl ID $replId, removing it")
          logger.error(msg)
          Seq(replId)
      }
    }
    brokenReplIds.foreach { replId =>
      subscribers.remove(replId)
    }
  }

  private[autologging] def publishEvent(
      replIdOpt: Option[String],
      sparkTableInfo: SparkTableInfo): Unit = synchronized {
    sparkTableInfo match {
      case SparkTableInfo(path, version, format) =>
        for ((replId, listener) <- getSubscribers) {
          if (replIdOpt.isEmpty || replId == replIdOpt.get) {
            try {
              listener.notify(path, version.getOrElse("unknown"), format.getOrElse("unknown"))
            } catch {
              case NonFatal(e) =>
                logger.error(s"Unable to forward event to listener with repl ID $replId. " +
                  s"Exception:\n${ExceptionUtils.serializeException(e)}")
            }
          }
        }
      case _ =>
    }
  }
}
