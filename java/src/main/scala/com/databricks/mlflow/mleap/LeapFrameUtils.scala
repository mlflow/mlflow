package com.databricks.mlflow.mleap;

import java.nio.charset.Charset

import ml.combust.mleap.json.DefaultFrameWriter
import ml.combust.mleap.runtime.frame.DefaultLeapFrame
import ml.combust.mleap.runtime.serialization.{BuiltinFormats, FrameReader, FrameWriter}

object LeapFrameUtils {
  
  val frameReader = FrameReader()
  val jsonCharset = Charset.forName("UTF-8")

  def getLeapFrameFromJson(inputJson : String) : DefaultLeapFrame = {
    val inputBytes = inputJson.getBytes(jsonCharset)
    frameReader.fromBytes(inputBytes).get
  }

  def getLeapFrameFromCsv(inputCsv : String) : DefaultLeapFrame = {
    throw new UnsupportedOperationException(
      "CSV to LeapFrame conversion has not been implemented yet!");
  }

  def getJsonFromLeapFrame(leapFrame : DefaultLeapFrame) : String = {
    val frameWriter = FrameWriter(leapFrame, BuiltinFormats.json)
    new String(frameWriter.toBytes().get)
  }

  def getCsvFromLeapFrame(leapFrame : DefaultLeapFrame) : String = {
    throw new UnsupportedOperationException(
      "LeapFrame to CSV conversion has not been implemented yet!");
  }

}
