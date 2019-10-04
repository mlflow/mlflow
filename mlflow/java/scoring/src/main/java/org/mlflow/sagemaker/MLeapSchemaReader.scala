package org.mlflow.sagemaker

import java.io.File
import java.nio.charset.Charset

import ml.combust.mleap.core.types.StructType
import ml.combust.mleap.json.JsonSupport._
import org.apache.commons.io.FileUtils
import spray.json._

class MLeapSchemaReader() {

  def fromFile(filePath: String) : StructType = {
    val json = FileUtils.readFileToString(new File(filePath), Charset.defaultCharset())
    json.parseJson.convertTo[StructType]
  }
}