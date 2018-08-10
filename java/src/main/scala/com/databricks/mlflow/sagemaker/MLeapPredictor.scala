package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.mleap.MLeapTransformerSchema 
import com.databricks.mlflow.mleap.LeapFrameUtils 

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.Transformer

import resource._

class MLeapPredictor(var modelPath : String, var inputSchemaPath : String) extends Predictor {
  val typedModelPath = "file:%s".format(modelPath)
  val bundle = (for(bundleFile <- managed(BundleFile(typedModelPath))) yield {
      bundleFile.loadMleapBundle().get
  }).opt.get
  val pipeline = bundle.root

  val inputSchema = MLeapTransformerSchema.fromFile(inputSchemaPath)

  def getPipeline() : Transformer = {
    this.pipeline
  }

  override def predict(inputFrame : DataFrame): DataFrame = {
    val frameJson = inputSchema.applyToPandasRecordJson(inputFrame.toJson())
    val leapFrame = LeapFrameUtils.getLeapFrameFromJson(frameJson)
    // TODO (Corey Zumar): Error handling
    val transformedFrame = pipeline.transform(leapFrame).get
    DataFrame.fromLeapFrame(transformedFrame)
  }

}
