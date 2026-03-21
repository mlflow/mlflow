package org.mlflow.spark.autologging

import java.lang.reflect.{Field, Method}

import scala.reflect.runtime.{universe => ru}
import org.slf4j.LoggerFactory

import scala.collection.mutable

private[autologging] object ReflectionUtils {
  private val logger = LoggerFactory.getLogger(getClass)
  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  /** Get Scala object by its fully-qualified name */
  def getScalaObjectByName(name: String): Any = {
    val module = rm.staticModule(name)
    val obj = rm.reflectModule(module)
    obj.instance
  }

  def getField(obj: Any, fieldName: String): Any = {
    var declaredFields: mutable.Buffer[Field] = obj.getClass.getDeclaredFields.toBuffer
    var superClass = obj.getClass.getSuperclass
    while (superClass != null) {
      declaredFields = declaredFields ++ superClass.getDeclaredFields
      superClass = superClass.getSuperclass
    }
    val field = declaredFields.find(_.getName == fieldName).getOrElse {
      throw new RuntimeException(s"Unable to get field '$fieldName' in object with class " +
        s"${obj.getClass.getName}. Available fields: " +
        s"[${declaredFields.map(_.getName).mkString(", ")}]")
    }
    field.setAccessible(true)
    field.get(obj)
  }

  /**
    * Call method with provided name on the specified object. The method name is assumed to be
    * unique
    */
  def callMethod(obj: Any, name: Any, args: Seq[Object]): Any = {
    var declaredMethods: mutable.Buffer[Method] = obj.getClass.getDeclaredMethods.toBuffer
    var superClass = obj.getClass.getSuperclass
    while (superClass != null) {
      declaredMethods = declaredMethods ++ superClass.getDeclaredMethods
      superClass = superClass.getSuperclass
    }
    val method = declaredMethods.find(_.getName == name).getOrElse(
      throw new RuntimeException(s"Unable to find method with name $name of object with class " +
        s"${obj.getClass.getName}. Available methods: " +
        s"[${declaredMethods.map(_.getName).mkString(", ")}]"))
    method.invoke(obj, args: _*)
  }

  def maybeCallMethod(obj: Any, name: Any, args: Seq[Object]): Option[Any] = {
    var declaredMethods: mutable.Buffer[Method] = obj.getClass.getDeclaredMethods.toBuffer
    var superClass = obj.getClass.getSuperclass
    while (superClass != null) {
      declaredMethods = declaredMethods ++ superClass.getDeclaredMethods
      superClass = superClass.getSuperclass
    }

    val methodOpt = declaredMethods.find(_.getName == name)

    methodOpt match {
      case Some(method) => Some(method.invoke(obj, args: _*))
      case None => None
    }
  }
}
