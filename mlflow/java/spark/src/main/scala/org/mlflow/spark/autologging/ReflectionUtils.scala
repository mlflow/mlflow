package org.mlflow.spark.autologging

import scala.reflect.runtime.{universe => ru}

private[autologging] object ReflectionUtils {

  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  /** Get Scala object by its fully-qualified name */
  def getScalaObjectByName(name: String): Any = {
    val module = rm.staticModule(name)
    val obj = rm.reflectModule(module)
    obj.instance
  }

  /** Get the specified field with specified name from the specified object */
  def getField(obj: Any, fieldName: String): Any = {
    val declaredFields = obj.getClass.getDeclaredFields
    val field = declaredFields.find(_.getName == fieldName).getOrElse {
      throw new RuntimeException(s"Unable to get field '$fieldName' in object with class " +
        s"${obj.getClass.getName}. Available fields: " +
        s"${declaredFields.map(_.getName).mkString(", ")}")
    }
    field.setAccessible(true)
    field.get(obj)
  }

  /**
    * Call method with provided name on the specified object. The method name is assumed to be
    * unique
    */
  def callMethod(obj: Any, name: Any, args: Seq[Object]): Any = {
    val declaredMethods = obj.getClass.getDeclaredMethods
    val method = declaredMethods.find(_.getName == name).getOrElse(
      throw new RuntimeException(s"Unable to find method with name $name of object with class " +
        s"${obj.getClass.getName}. Available methods: " +
        s"${declaredMethods.map(_.getName).mkString(", ")}"))
    method.invoke(obj, args: _*)
  }
}
