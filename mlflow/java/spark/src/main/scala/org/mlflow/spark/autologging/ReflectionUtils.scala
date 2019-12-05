package org.mlflow.spark.autologging

import scala.reflect.runtime.{universe => ru}

import java.lang.reflect.Method

object ReflectionUtils {

  private val rm = ru.runtimeMirror(getClass.getClassLoader)
  def getMethod(obj: Object, name: String): Method = {
    obj.getClass.getDeclaredMethods.find(_.getName == name).getOrElse(
      throw new RuntimeException(s"Unable to find method $name for instance of " +
        s"${obj.getClass.getName}"))
  }

  /** Get Scala object by its fully-qualified name */
  def getScalaObjectByName(name: String): Any = {
    val module = rm.staticModule(name)
    val obj = rm.reflectModule(module)
    obj.instance
  }

  def getField(obj: Any, fieldName: String): Any = {
    val field = obj.getClass.getDeclaredField(fieldName)
    field.get(obj)
  }

  /** Call method with provided name (assumed to be unique) on specified object */
  def callMethod(obj: Any, name: Any, args: Seq[Object]): Any = {
    val method = obj.getClass.getDeclaredMethods.find(_.getName == name).getOrElse(throw new RuntimeException("Uh oh"))
    method.invoke(obj, args: _*)
  }
}
