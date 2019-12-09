package org.mlflow.spark.autologging

import java.nio.file.{Files, Path, Paths}
import java.util.UUID

import org.apache.spark.mlflow.MlflowSparkAutologgingTestUtils
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.mockito.Matchers.any
import org.mockito.Mockito._
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach}
import org.scalatest.mockito.MockitoSugar
import org.scalatest.FunSuite
import org.scalatest.Matchers

import scala.collection.mutable.ArrayBuffer

private[autologging] class MockSubscriber extends SparkDataSourceEventSubscriber {
  private val uuid: String = UUID.randomUUID().toString
  override def replId: String = {
    uuid
  }

  override def notify(path: String, version: String, format: String): Unit = {
  }

  override def ping(): Unit = {}
}

private[autologging] class BrokenSubscriber extends MockSubscriber {
  override def ping(): Unit = {
    throw new RuntimeException("Oh no, failing ping!")
  }
}

class SparkAutologgingSuite extends FunSuite with Matchers with BeforeAndAfterAll
  with BeforeAndAfterEach {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("MLflow Spark Autologging Tests")
    .config("spark.master", "local")
    .getOrCreate()

  var tempDir: Path  = _
  var formatToTablePath: Map[String, String] = _
  var deltaTablePath: String = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    // Generate dummy data & write it in various formats (Delta, CSV, JSON)
    val rows = Seq(
      Row(8, "bat"),
      Row(64, "mouse"),
      Row(-27, "horse")
    )
    val schema = List(
      StructField("number", IntegerType),
      StructField("word", StringType)
    )
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      StructType(schema)
    )
    tempDir = Files.createTempDirectory(this.getClass.getName)
    deltaTablePath = Paths.get(tempDir.toString, "delta").toString
    formatToTablePath = Seq( "csv", "parquet", "json" /*, delta */).map { format =>
      format -> Paths.get(tempDir.toString, format).toString
    }.toMap

    formatToTablePath.foreach { case (format, tablePath) =>
      df.write.option("header", "true").format(format).save(tablePath)
    }
  }

  override def afterAll(): Unit = {
    super.afterAll()
    // TODO find replacement for this that handles deleting non-empty dirs
    // Files.delete(tempDir)
  }

  override def beforeEach(): Unit = {
    super.beforeEach()
    SparkDataSourceEventPublisher.init()
  }

  override def afterEach(): Unit = {
    SparkDataSourceEventPublisher.stop()
    super.afterEach()
  }

  private def getFileUri(absolutePath: String): String = {
    s"${Paths.get("file:", absolutePath).toString}"
  }

  test("SparkDataSourceEventPublisher can be idempotently initialized & stopped within " +
    "single thread") {
    // We expect a listener to already be created by calling init() in beforeEach
    val listeners0 = MlflowSparkAutologgingTestUtils.getListeners(spark)
    assert(listeners0.length == 1)
    val listener0 = listeners0.head
    // Call init() again, verify listener is unchanged
    SparkDataSourceEventPublisher.init()
    val listeners1 = MlflowSparkAutologgingTestUtils.getListeners(spark)
    assert(listeners1.length == 1)
    val listener1 = listeners1.head
    assert(listener0 == listener1)
    // Call stop() multiple times
    SparkDataSourceEventPublisher.stop()
    assert(MlflowSparkAutologgingTestUtils.getListeners(spark).isEmpty)
    SparkDataSourceEventPublisher.stop()
    assert(MlflowSparkAutologgingTestUtils.getListeners(spark).isEmpty)
    // Call init() after stop(), verify that we create a new listener
    SparkDataSourceEventPublisher.init()
    val listeners2 = MlflowSparkAutologgingTestUtils.getListeners(spark)
    assert(listeners2.length == 1)
    val listener2 = listeners2.head
    assert(listener2 != listener1)
  }

  test("SparkDataSourceEventPublisher triggers publishEvent with appropriate arguments " +
    "when reading datasources corresponding to different formats") {
      val formatToTestDFs = formatToTablePath.map { case (format, tablePath) =>
        val baseDf = spark.read.format(format).option("inferSchema", "true")
          .option("header", "true").load(tablePath)
        format -> Seq(
          baseDf,
          baseDf.filter("number > 0"),
          baseDf.select("number"),
          baseDf.limit(2),
          baseDf.filter("number > 0").select("number").limit(2)
        )
      }

        formatToTestDFs.foreach { case (format, dfs) =>
          dfs.foreach { df =>
            df.printSchema()
            SparkDataSourceEventPublisher.init()
            val subscriber = spy(new MockSubscriber())
            SparkDataSourceEventPublisher.register(subscriber)
            assert(SparkDataSourceEventPublisher.subscribers.size == 1)
            // Read DF
            df.collect()
            // Verify events logged
            Thread.sleep(1000)
            val tablePath = formatToTablePath(format)
            val expectedPath = getFileUri(tablePath)
            verify(subscriber, times(1)).notify(any(), any(), any())
            verify(subscriber, times(1)).notify(expectedPath, "unknown", format)
            SparkDataSourceEventPublisher.stop()
          }
        }
  }

  test("SparkDataSourceEventPublisher triggers publishEvent with appropriate arguments " +
    "when reading a JOIN of two tables") {
    val formats = formatToTablePath.keys
    val leftFormat = formats.head
    val rightFormat = formats.last
    val leftPath = formatToTablePath(leftFormat)
    val rightPath = formatToTablePath(rightFormat)
    val leftDf = spark.read.format(leftFormat).load(leftPath)
    val rightDf = spark.read.format(rightFormat).load(rightPath)
    SparkDataSourceEventPublisher.init()
    val subscriber = spy(new MockSubscriber())
    SparkDataSourceEventPublisher.register(subscriber)
    leftDf.join(rightDf).collect()
    // Sleep a second to let the SparkListener trigger read
    Thread.sleep(1000)
    verify(subscriber, times(2)).notify(any(), any(), any())
    verify(subscriber, times(1)).notify(getFileUri(leftPath), "unknown", leftFormat)
    verify(subscriber, times(1)).notify(getFileUri(rightPath), "unknown", rightFormat)
  }

  test("SparkDataSourceEventPublisher publishes to subscribers in order of registration") {
    val notifyOrder: ArrayBuffer[Int] = new ArrayBuffer[Int]()
    class SaveNotifyOrderSubscriber(idx: Int) extends MockSubscriber {
      override def notify(path: String, version: String, format: String): Unit = {
        notifyOrder.append(idx)
      }
    }
    val (format, path) = formatToTablePath.head
    val df = spark.read.format(format).load(path)
    val subscribers = 0.until(3).map(idx => new SaveNotifyOrderSubscriber(idx))
    subscribers.foreach(SparkDataSourceEventPublisher.register)
    df.collect()
    Thread.sleep(1000)
    assert(notifyOrder == Seq(0, 1, 2))
  }


  test("SparkDataSourceEventPublisher can publish to working subscribers even when " +
    "others are broken") {
    // Reinitialize publisher with very large GC interval
    SparkDataSourceEventPublisher.stop()
    SparkDataSourceEventPublisher.init(gcDeadSubscribersIntervalSec = 10000)
    // Register broken subscribers & a working subscriber, verify we can publish to the working
    // subscriber despite the broken ones
    val (format, path) = formatToTablePath.head
    val df = spark.read.format(format).load(path)
    SparkDataSourceEventPublisher.register(new BrokenSubscriber())
    val subscriber = spy(new MockSubscriber())
    SparkDataSourceEventPublisher.register(subscriber)
    SparkDataSourceEventPublisher.register(new BrokenSubscriber())
    df.collect()
    Thread.sleep(1000)
    verify(subscriber, times(1)).notify(any(), any(), any())
    verify(subscriber, times(1)).notify(getFileUri(path), "unknown", format)
  }

  test("SparkDataSourceEventPublisher correctly unregisters broken subscribers") {
    SparkDataSourceEventPublisher.register(new BrokenSubscriber())
    Thread.sleep(2000)
    assert(SparkDataSourceEventPublisher.subscribers.isEmpty)
  }

  test("Subscriber registration fails if init() not called") {
    SparkDataSourceEventPublisher.stop()
    intercept[RuntimeException] {
      SparkDataSourceEventPublisher.register(new MockSubscriber())
    }
  }

}
