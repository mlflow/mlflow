package org.mlflow.spark.autologging

import org.scalatest.FunSuite

object TestObject {
  def myMethod: String = "hi"
}

abstract class TestAbstractClass {
  protected def addNumbers(x: Int, y: Int): Int = x + y
  protected val myProtectedVal: Int = 5
}

class RealClass extends TestAbstractClass {
  private val myField: String = "myCoolVal"
  def subclassMethod(x: Int): Int = x * x
}

class ReflectionUtilsSuite extends FunSuite {

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
}
