import styles from "./note.module.css";

export default function Note(): JSX.Element {
  return (
    <div className={styles.note}>
      <p className={styles.title}>Note</p>
      <p>
        When your input is a dictionary format that represents request payload,
        it can also include the parameters like{" "}
        <code className="docutils literal notranslate">
          <span className="pre">max_tokens</span>
        </code>
        . If there are overlapping parameters in both the{" "}
        <code className="docutils literal notranslate">
          <span className="pre">inference_params</span>
        </code>{" "}
        and the input data, the values in the{" "}
        <code className="docutils literal notranslate">
          <span className="pre">inference_params</span>
        </code>{" "}
        will take precedence.
      </p>
    </div>
  );
}
