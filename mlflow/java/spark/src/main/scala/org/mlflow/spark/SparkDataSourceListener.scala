package org.mlflow.spark.autologging

import java.util.concurrent._

import org.apache.spark.sql.SparkAutologgingUtils

import scala.util.control.NonFatal
import scala.collection.JavaConverters._
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.apache.spark.sql.execution.SQLExecution
import org.apache.spark.SparkContext
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}


import scala.collection.{immutable, mutable}
import scala.util.control.NonFatal
import scala.collection.JavaConverters._

import org.apache.spark.SparkContext


import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui._
import org.apache.spark.sql.execution.{GenerateExec, SparkPlan, SQLExecution}
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}

// TODO replace this with OSS logging library
import  org.apache.spark.sql.delta.{DeltaTable, DeltaFullTable}

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, LeafNode}

import java.time.format.DateTimeFormatter
import java.time.LocalDateTime


// TODO replace this with OSS logging library

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

  def replId: String

}

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
  var subscribers: Map[String, SparkDataSourceEventSubscriber] = Map()


  def spark: SparkSession = {
    SparkSession.builder.getOrCreate()
  }

  // Initialize Spark listener that pulls Delta query plan information & bubbles it up to registered
  // Python subscribers
  def init(): Unit = synchronized {
    if (sparkQueryListener == null) {

      // Get SparkContext & determine if REPL id is set - if not, then we log irrespective of repl
      // ID, but if so, we log conditionally on repl ID
      val sc = SparkContext.getOrCreate()
      val replId = Option(sc.getLocalProperty("spark.databricks.replId"))
      sparkQueryListener = replId match {
        case None => new SparkDataSourceListener()
        case Some(replId) => new DatabricksSparkDataSourceListener()
      }

      // TODO: try-catch this & reset listener to null if it fails?
      spark.sparkContext.addSparkListener(sparkQueryListener)
      // Schedule regular cleanup of detached subSparkListenerInterfacescribers, e.g. those associated with detached notebooks
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

  def publishEvent(replIdOpt: Option[String], sparkTableInfo: SparkTableInfo): Unit = synchronized {
    sparkTableInfo match {
      case SparkTableInfo(path, version, format) =>
        val time = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH:mm:ss").format(LocalDateTime.now)
        println(s"Notifying ${subscribers.size} listeners about read to ${path}, ${version}, " +
          s"time ${time}")

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

case class SparkTableInfo(path: String, versionOpt: Option[String], formatOpt: Option[String])


class SparkDataSourceListener extends SparkListener {
  protected def getLeafNodes(lp: LogicalPlan): Seq[LogicalPlan] = {
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

  // Get SparkTableInfo of info to log from leaf node of a query plan
  protected def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {
    leafNode match {
      case DeltaTable(tahoeFileIndex) =>
        val path = tahoeFileIndex.path.toString
        val versionOpt = Option(tahoeFileIndex.tableVersion).map(_.toString)
        Option(SparkTableInfo(path, versionOpt, Option("delta")))
      case DeltaFullTable(tahoeFileIndex) =>
        val path = tahoeFileIndex.path.toString
        val versionOpt = Option(tahoeFileIndex.tableVersion).map(_.toString)
        Option(SparkTableInfo(path, versionOpt, Option("delta")))
      case LogicalRelation(HadoopFsRelation(index, _, _, _, _, _), _, _, _) =>
        val path: String = index.rootPaths.headOption.map(_.toString).getOrElse("unknown")
        Option(SparkTableInfo(path, None, None))
      case other =>
        None
    }
  }

  protected def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val qe = SparkAutologgingUtils.getQueryExecution(event)
    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      val tableInfosToLog = leafNodes.flatMap(getTableInfoToLog)
      tableInfosToLog.foreach(tableInfo => SparkDataSourceEventPublisher.publishEvent(None,
        tableInfo))
    }
  }


  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionEnd =>
        onSQLExecutionEnd(e)
      case _ =>
    }
  }
}

/**
  * SparkListener implementation that attempts to extract Delta table information & notify the PySpark process
  * TODO: maybe pull query-plan-parsing logic out so that we can later add autologging for users of the Java client
  * as well
  */
class DatabricksSparkDataSourceListener() extends SparkDataSourceListener {
  // Order of callbacks is onSQLExecutionStart, onJobStart, onSQLExecutionEnd
  // So we can figure out where to log table infos in onJobStart (i.e. find assoc REPL),
  // and log them & remove them in onSQLExecutionEnd
  private val executionIdToReplId = new java.util.concurrent.ConcurrentHashMap[Long, String]()

  override def onJobStart(event: SparkListenerJobStart): Unit = {
    // Find corresponding execution, and
    // If found, check if we have an associated active SQLExecutionAdvisor, and set its job group.
    // This is needed, as in onSQLExecutionStart we don't know the SQLExecution's job group.
    val properties: Map[String, String] = Option(event.properties).map(_.asScala.toMap).getOrElse(Map.empty)
    val executionIdOpt = properties.get(SQLExecution.EXECUTION_ID_KEY).map(_.toLong)
    if (executionIdOpt.isEmpty) {
      return
    }
    val executionId = executionIdOpt.get
    val replIdOpt = properties.get("spark.databricks.replId")
    replIdOpt.foreach { replId =>
      executionIdToReplId.put(executionId, replId)
    }
  }

  override protected def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    val qe = SparkAutologgingUtils.getQueryExecution(event)
    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      val tableInfosToLog = leafNodes.flatMap(getTableInfoToLog)
      val executionId = event.executionId
      // Remove the executionId -> replId mapping if it exists. remove() returns null if no key
      // was found, so wrap it in an option & log the table infos only if we can find a
      // corresponding repl ID
      Option(executionIdToReplId.remove(executionId)).foreach { replId =>
        tableInfosToLog.foreach(tableInfo => SparkDataSourceEventPublisher.publishEvent(
          Option(replId), tableInfo))
      }
    }
  }
}