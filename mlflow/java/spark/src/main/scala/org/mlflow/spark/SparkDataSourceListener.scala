package org.mlflow.spark.autologging

import java.util.concurrent._

import scala.util.control.NonFatal

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import org.apache.spark.sql.SparkAutologgingUtils
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}

// TODO replace this with OSS logging library
import  org.apache.spark.sql.delta.{DeltaTable, DeltaFullTable}

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, LeafNode}

import java.time.format.DateTimeFormatter
import java.time.LocalDateTime

import py4j.Py4JException

/* You can add arbitrary methods here,
 * as long as these match corresponding Python interface
 */
trait SparkDataSourceEventSubscriber {
  /* This will be implemented by a Python class.
   * You can of course use more specific types,
   * for example here String => Unit */
  def notify(path: String, version: String, format: String): Any

  def ping(): Unit
}


object SparkDataSourceEventPublisher {
  val logger = LoggerFactory.getLogger(getClass)

  private[autologging] var sparkQueryListener: SparkDataSourceListener = _
  var subscribers: Map[String, SparkDataSourceEventSubscriber] = Map()


  def spark: SparkSession = {
    SparkSession.builder.getOrCreate()
  }

  // Initialize Spark listener that pulls Delta query plan information & bubbles it up to registered
  // Python subscribers
  def init(): Unit = synchronized {
    if (sparkQueryListener == null) {
      sparkQueryListener = new SparkDataSourceListener()
      // TODO: try-catch this & reset listener to null if it fails?
      spark.sparkContext.addSparkListener(sparkQueryListener)
      // Schedule regular cleanup of detached subscribers, e.g. those associated with detached notebooks
      val ex = new ScheduledThreadPoolExecutor(1)
      val task = new Runnable {
        def run(): Unit = {
          unregisterBrokenSubscribers()
        }
      }
      ex.scheduleAtFixedRate(task, 1, 1, TimeUnit.SECONDS)
    }
  }

  def stop(): Unit = synchronized {
    if (sparkQueryListener != null) {
      spark.sparkContext.removeSparkListener(sparkQueryListener)
      sparkQueryListener = null
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

  def unregister(uuid: String): Unit = synchronized {
    if (sparkQueryListener == null) {
      throw new RuntimeException("Please call init() before attempting to unregister a subscriber")
    }
    this.synchronized {
      subscribers = subscribers - uuid
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
            s"Please report this error at https://github.com/mlflow/mlflow/issues, along with" +
            s"the following stacktrace:\n$e")
          Seq(uuid)
      }
    }
    brokenUuids.foreach(unregister)
  }

  def notifyAll(path: String, version: Option[String], format: Option[String]): Unit = synchronized {
    val time = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss").format(LocalDateTime.now)
    println(s"Notifying ${subscribers.size} listeners about read to ${path}, ${version}, " +
      s"time ${time}")

    for ((uuid, listener) <- subscribers) {
      try {
        listener.notify(path, version.getOrElse("unknown"), format.getOrElse("unknown"))
      } catch {
        case e: Py4JException =>
          logger.error(s"Unable to forward event to listener with UUID ${uuid}.")
      }
    }
  }
}

/**
  * SparkListener implementation that attempts to extract Delta table information & notify the PySpark process
  * TODO: maybe pull query-plan-parsing logic out so that we can later add autologging for users of the Java client
  * as well
  */
class SparkDataSourceListener() extends SparkListener {
  private def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
    if (lp == null) {
      return Seq.empty
    }
    if (lp.isInstanceOf[LeafNode]) {
      Seq(lp)
    } else {
      lp.children.flatMap { child =>
        child match {
          case l: LeafNode =>
            Seq(l)
          case other: LogicalPlan => getLeafNodes(other)
        }
      }
    }
  }
  // Send a notification to all listeners managed by Manager if the leafNode corresponds to a
  // Delta table read. TODO: extend this method to detect reads of Spark datasources in other formats,
  // e.g. JSON or CSV.
  private def notifyIfSparkTableRead(leafNode: LogicalPlan): Unit = {
    leafNode match {
      case DeltaTable(tahoeFileIndex) =>
        val path = tahoeFileIndex.path.toString
        val versionOpt = Option(tahoeFileIndex.tableVersion).map(_.toString)
        SparkDataSourceEventPublisher.notifyAll(path, versionOpt, Option("delta"))
      case DeltaFullTable(tahoeFileIndex) =>
        val path = tahoeFileIndex.path.toString
        val versionOpt = Option(tahoeFileIndex.tableVersion).map(_.toString)
        SparkDataSourceEventPublisher.notifyAll(path, versionOpt, Option("delta"))
      case LogicalRelation(HadoopFsRelation(index, _, _, _, _, _), _, _, _) =>
        val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
        SparkDataSourceEventPublisher.notifyAll(path, None, None)
      case other =>
        None
    }
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        val qe = SparkAutologgingUtils.getQueryExecution(e)
        if (qe != null) {
          val children = getLeafNodes(qe.analyzed)
          children.foreach { child =>
            notifyIfSparkTableRead(child)
          }
        }
      case _ =>
    }
  }
}
