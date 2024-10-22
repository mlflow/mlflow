import styles from "./hint.module.css";

export default function Hint(): JSX.Element {
  return (
    <div className={styles.hint}>
      <p className={styles.title}>Hint</p>
      <p>
        When you want to use an endpoint <strong>not</strong> hosted by an
        MLflow AI Gateway or Databricks, you can create a custom Python function
        following the{" "}
        <a className="reference internal" href="#llm-eval-custom-function">
          <span className="std std-ref">Evaluating with a Custom Function</span>
        </a>{" "}
        guide and use it as the{" "}
        <code className="docutils literal notranslate">
          <span className="pre">model</span>
        </code>{" "}
        argument.
      </p>
    </div>
  );
}
