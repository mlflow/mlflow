import CodeBlock from "@theme/CodeBlock";
import styles from "./styles.module.css";

export const NotebookCodeCell = ({ children, executionCount }): JSX.Element => (
  <div
    style={{
      display: "flex",
      flexDirection: "row",
      marginTop: "var(--padding-md)",
      width: "100%",
    }}
  >
    <div style={{ width: "2rem", flexShrink: 0, fontSize: "0.8rem" }}>
      {`[${executionCount}]`}
    </div>
    <div style={{ flexGrow: 1, minWidth: 0 }}>
      <CodeBlock className={styles.codeBlock} language="python">
        {children}
      </CodeBlock>
    </div>
  </div>
);
