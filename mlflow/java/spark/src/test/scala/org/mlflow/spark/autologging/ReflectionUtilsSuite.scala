package org.mlflow.spark.autologging

import org.scalatest.funsuite.AnyFunSuite

object TestObject {
  def myMethod: String = "hi"
}

object TestFileIndex {
  def version: String = "1.0"
}

abstract class TestAbstractClass {
  protected def addNumbers(x: Int, y: Int): Int = x + y
  protected val myProtectedVal: Int = 5
}

class RealClass extends TestAbstractClass {
  private val myField: String = "myCoolVal"
  def subclassMethod(x: Int): Int = x * x
}

class ReflectionUtilsSuite extends AnyFunSuite {

  test("Can get private & protected fields of an object via reflection") {
    val obj = new RealClass()
    val field0 = ReflectionUtils.getField(obj, "myField").asInstanceOf[String]
    assert(field0 == "myCoolVal")
    val field1 = ReflectionUtils.getField(obj, "myProtectedVal").asInstanceOf[Int]
    assert(field1 == 5)
  }

  test("Can call methods via reflection") {
    val obj = new RealClass()
    val args0: Seq[Object] = Seq[Integer](3)
    val res0 = ReflectionUtils.callMethod(obj, "subclassMethod", args0).asInstanceOf[Int]
    assert(res0 == 9)
    val args1: Seq[Object] = Seq[Integer](5, 6)
    val res1 = ReflectionUtils.callMethod(obj, "addNumbers", args1).asInstanceOf[Int]
    assert(res1 == 11)
  }

  test("Can get Scala object and call methods via reflection") {
    val obj = ReflectionUtils.getScalaObjectByName("org.mlflow.spark.autologging.TestObject")
    val res = ReflectionUtils.callMethod(obj, "myMethod", Seq.empty).asInstanceOf[String]
    assert(res == "hi")
  }

  test("maybeCallMethod None if method not found") {
    val obj = new RealClass()
    val res = ReflectionUtils.maybeCallMethod(obj, "nonExistentMethod", Seq.empty)

    assert (res.isEmpty)
  }

  test("maybeCallMethod invokes the method if the method is found") {
    val obj = ReflectionUtils.getScalaObjectByName("org.mlflow.spark.autologging.TestObject")
    val res0 = ReflectionUtils.maybeCallMethod(obj, "myMethod", Seq.empty).getOrElse("")
    assert(res0 == "hi")
  }

  test("chaining maybeCallMethod works") {
    val fileIndex = ReflectionUtils.getScalaObjectByName("org.mlflow.spark.autologging.TestFileIndex")

    val versionOpt0 = ReflectionUtils.maybeCallMethod(fileIndex, "version", Seq.empty).orElse(
      Option("second thing")
    ).map(_.toString)
    assert(versionOpt0 == Some("1.0"))

    // if only the second method exists, return it
    val versionOpt1 = ReflectionUtils.maybeCallMethod(fileIndex, "tableVersion", Seq.empty).orElse(
      ReflectionUtils.maybeCallMethod(fileIndex, "version", Seq.empty)
    ).map(_.toString)
    assert(versionOpt1 == Some("1.0"))

    // if both don't exist, just return None
    val versionOpt2 = ReflectionUtils.maybeCallMethod(fileIndex, "tableVersion", Seq.empty).orElse(
      ReflectionUtils.maybeCallMethod(fileIndex, "anotherTableVersion", Seq.empty)
    ).map(_.toString)
    assert(versionOpt2 == None)
  }
}
