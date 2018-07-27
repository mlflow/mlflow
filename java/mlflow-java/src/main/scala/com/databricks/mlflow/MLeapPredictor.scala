package com.databricks.mlflow.mleap;

import com.databricks.mlflow.models.ScalaModel

import java.nio.charset.Charset

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.serialization.FrameReader

import resource._

class MLeapPredictor(var modelPath : String) extends ScalaModel {
  val typedModelPath = "file:%s".format(modelPath)
  val bundle = (for(bundleFile <- managed(BundleFile(typedModelPath))) yield {
      bundleFile.loadMleapBundle().get
  }).opt.get

  val pipeline = bundle.root
  val frameReader = FrameReader()
  val jsonCharset = Charset.forName("UTF-8")

  // override def predict(input : InputBean): String = {
  override def predict(input : String): String = {
    // val inputJson = input.getInputJson()
    // val inputBytes = inputJson.getBytes(jsonCharset)
    // val deserializedFrame = frameReader.fromBytes(inputBytes).get
    //
    // // TODO (Corey Zumar): Error handling
    // val transformedFrame = pipeline.transform(deserializedFrame).get
    // val output = new String(transformedFrame.writer().toBytes().get);
    // output
    "BLAH"
  }

}
