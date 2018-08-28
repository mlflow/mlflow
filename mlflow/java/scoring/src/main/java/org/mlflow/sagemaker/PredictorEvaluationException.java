package org.mlflow.sagemaker;

/**
 * An exception that is intended to be thrown when a {@link org.mlflow.sagemaker.Predictor} object
 * encounters an error during inference.
 */
public class PredictorEvaluationException extends Exception {
  /**
   * Constructs a {@link PredictorEvaluationException}
   *
   * @param message The user-readable error message associated with this exception
   */
  public PredictorEvaluationException(String message) {
    super(message);
  }

  /**
   * Constructs a {@link PredictorEvaluationException}
   *
   * @param message The user-readable error message associated with this exception
   * @param cause The cause of the exception
   */
  public PredictorEvaluationException(String message, Throwable cause) {
    super(message, cause);
  }
}
