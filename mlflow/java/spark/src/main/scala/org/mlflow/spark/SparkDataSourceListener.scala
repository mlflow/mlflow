package org.mlflow.spark.autologging

import java.util.concurrent._

import org.apache.spark.sql.SparkAutologgingUtils

import scala.util.control.NonFatal
import scala.collection.JavaConverters._
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.apache.spark.sql.execution.{GenerateExec, QueryExecution, SQLExecution, SparkPlan}
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
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.delta.{DeltaFullTable, DeltaTable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.plans.logical.{LeafNode, LogicalPlan}
import java.time.format.DateTimeFormatter
import java.time.LocalDateTime
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
  val ex = new ScheduledThreadPoolExecutor(1)
  var subscribers: Map[String, SparkDataSourceEventSubscriber] = Map()
  var scheduledTask: ScheduledFuture[_] = null

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
      val listener = replId match {
        case None => new SparkDataSourceListener()
        case Some(replId) => new DatabricksSparkDataSourceListener()
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
  // So we can get the table infos to log in onSQLExecutionStart, figure out where to log them in onJobStart,
  // and remove them in onSQLExecutionEnd

  // A QueryExecution has many JobIDs via associatedJobs = mutable.Set[Int]()
  // A SparkListenerJobStart event gives you a mapping from JobIds to repl IDs (so you know the repl associated with each job)
  // So should be able to capture the first mapping on Job start, then on query execution end look up
  // the repls associated with each Spark job that was completed, and notify those repls
  private val executionIdToTableInfos = mutable.Map[Long, Seq[SparkTableInfo]]()

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

  // Get SparkTableInfo of info to log from leaf node of a query plan
  private def getTableInfoToLog(leafNode: LogicalPlan): Option[SparkTableInfo] = {

    val deltaTableObj = Class.forName("com.databricks.sql.transaction.tahoe.DeltaTable$")

    leafNode match {
      case DeltaTable(tahoeFileIndex) =>
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

  private def addTableInfos(executionId: Long, tableInfos: Seq[SparkTableInfo]): Unit = synchronized {
    val tableInfosOpt = executionIdToTableInfos.get(executionId)
    if (tableInfosOpt.isDefined) {
      throw new RuntimeException(
        s"Unexpected error trying to associate " +
          s"execution ID ${executionId} -> table infos. Found existing table infos.")
    } else {
      executionIdToTableInfos(executionId) = tableInfos
    }
  }

  private def removeExecutionIdToTableInfos(executionId: Long): Unit = synchronized {
    executionIdToTableInfos.remove(executionId)
  }

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
      executionIdToTableInfos.get(executionId).map { tableInfosToLog =>
        tableInfosToLog.map(tableInfo => SparkDataSourceEventPublisher.publishEvent(Option(replId), tableInfo))
      }
    }
  }

  // Populate a map of execution ID to list of table infos to log under
  private def onSQLExecutionStart(event: SparkListenerSQLExecutionStart): Unit = {
    val qe: QueryExecution = event.getClass.getDeclaredFields.find(_.getName == "qe")
        .map(_.get(event).asInstanceOf[QueryExecution]).getOrElse {
      throw new RuntimeException("Unable to get QueryExecution field")
    }

    if (qe != null) {
      val leafNodes = getLeafNodes(qe.analyzed)
      val tableInfosToLog = leafNodes.flatMap(getTableInfoToLog)
      addTableInfos(event.executionId, tableInfosToLog)
    }
  }

  private def onSQLExecutionEnd(event: SparkListenerSQLExecutionEnd): Unit = {
    removeExecutionIdToTableInfos(event.executionId)
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = {
    event match {
      case e: SparkListenerSQLExecutionStart =>
        onSQLExecutionStart(e)
      case e: SparkListenerSQLExecutionEnd =>
        onSQLExecutionEnd(e)
      case _ =>
    }
  }
}