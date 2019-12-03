package org.mlflow.spark.autologging

import java.nio.file.{Files, Path, Paths}

import org.apache.spark.mlflow.MlflowSparkAutologgingTestUtils
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.mockito.Matchers.any

import scala.collection._
import org.mockito.Mockito._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.mockito.MockitoSugar

/*
Here's an example of a FunSuite with Matchers mixed in:
*/
import org.scalatest.FunSuite
import org.scalatest.Matchers

class SparkAutologgingSuite extends FunSuite with Matchers with BeforeAndAfterAll {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("MLflow Spark Autologging Tests")
    .config("spark.master", "local")
    .getOrCreate()

  var tempDir: Path  = _
  var formatToTablePath: Map[String, String] = _
  var deltaTablePath: String = _

  override def beforeAll(): Unit = {
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
    formatToTablePath = Seq( /* "delta", */ "csv", "json", "parquet").map { format =>
      format -> Paths.get(tempDir.toString, format).toString
    }.toMap

    formatToTablePath.foreach { case (format, tablePath) =>
      df.write.format(format).save(tablePath)
    }
  }

  override def afterAll(): Unit = {
    Files.delete(tempDir)
  }

  test("SparkDataSourceEventPublisher can be idempotently initialized & stopped within " +
    "single thread") {
    SparkDataSourceEventPublisher.init()
    val listeners0 = MlflowSparkAutologgingTestUtils.getListeners(spark)
    assert(listeners0.length == 1)
    val listener0 = listeners0.head
    SparkDataSourceEventPublisher.init()
    val listeners1 = MlflowSparkAutologgingTestUtils.getListeners(spark)
    assert(listeners1.length == 1)
    val listener1 = listeners1.head
    assert(listener0 == listener1)
    SparkDataSourceEventPublisher.stop()
    assert(MlflowSparkAutologgingTestUtils.getListeners(spark).isEmpty)
    SparkDataSourceEventPublisher.stop()
    assert(MlflowSparkAutologgingTestUtils.getListeners(spark).isEmpty)
  }

  test("SparkDataSourceInitializer triggers publishEvent with appropriate arguments " +
    "when reading datasources corresponding to different formats") {
    class MockSparkDatasourceEventPublisher extends SparkDataSourceEventPublisherImpl {

    }
    val publisher = spy(new MockSparkDatasourceEventPublisher())
    publisher.init()
    formatToTablePath.foreach { case (format, tablePath) =>
      spark.read.format(format).load(tablePath)
    }
    verify(publisher, times(formatToTablePath.size)).publishEvent(any(), any())

    (formatToTablePath -- Seq("delta")).values.foreach { tablePath =>
      verify(publisher, times(1)).publishEvent(any(), SparkTableInfo(tablePath, None, None))
    }
//    verify(publisher, times(1)).publishEvent(
//      any(), SparkTableInfo(deltaTablePath, Option("0"), Option("delta")))
  }

}
