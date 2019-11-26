package org.mlflow.spark.autologging

import scala.collection._
import org.scalatest.Assertions
import org.junit.Test

/*
Here's an example of a FunSuite with Matchers mixed in:
*/
import org.scalatest.FunSuite
import org.scalatest.Matchers

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
@RunWith(classOf[JUnitRunner])
class ListSuite extends FunSuite with Matchers {

  test("An empty list should be empty") {
    List() should be ('empty)
  }
}
