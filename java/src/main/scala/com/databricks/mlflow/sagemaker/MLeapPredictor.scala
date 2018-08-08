package com.databricks.mlflow.sagemaker;

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.Transformer

import resource._

class MLeapPredictor(var modelPath : String) extends Predictor {
  val typedModelPath = "file:%s".format(modelPath)
  val bundle = (for(bundleFile <- managed(BundleFile(typedModelPath))) yield {
      bundleFile.loadMleapBundle().get
  }).opt.get

  val pipeline = bundle.root

  def getPipeline() : Transformer = {
    this.pipeline
  }

  override def predict(inputFrame : DataFrame): DataFrame = {
    // TODO (Corey Zumar): Error handling
    val transformedFrame = pipeline.transform(inputFrame.getLeapFrame()).get
    new DataFrame(transformedFrame)
  }

}
