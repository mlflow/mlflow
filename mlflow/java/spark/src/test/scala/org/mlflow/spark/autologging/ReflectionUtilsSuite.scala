package org.mlflow.spark.autologging

import org.scalatest.FunSuite

object TestObject {
  def myMethod: String = "hi"
}

abstract class TestAbstractClass {
  def addNumbers(x: Int, y: Int): Int = x + y
}

class RealClass extends TestAbstractClass {
  private val myField: String = "myCoolVal"
  def subclassMethod(x: Int): Int = x * x
}

class ReflectionUtilsSuite extends FunSuite {

  test("Can get private field of an object via reflection") {
    val obj = new RealClass()
    val field = ReflectionUtils.getField(obj, "myField").asInstanceOf[String]
    assert(field == "myCoolVal")
  }

  test("Can call methods via reflection") {
    val obj = new RealClass()
    val args0: Seq[Object] = Seq[Integer](3)
    val res0 = ReflectionUtils.callMethod(obj, "subclassMethod", args0).asInstanceOf[Int]
    assert(res0 == 9)
  }

  test("Can get Scala object and call methods via reflection") {
    val obj = ReflectionUtils.getScalaObjectByName("org.mlflow.spark.autologging.TestObject")
    val res = ReflectionUtils.callMethod(obj, "myMethod", Seq.empty).asInstanceOf[String]
    assert(res == "hi")
  }
}
