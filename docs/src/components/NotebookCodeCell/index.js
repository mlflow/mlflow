import CodeBlock from '@theme/CodeBlock';
import styles from './styles.module.css';
export var NotebookCodeCell = function (_a) {
    var children = _a.children, executionCount = _a.executionCount;
    return (<div style={{
            flexGrow: 1,
            minWidth: 0,
            marginTop: 'var(--padding-md)',
            width: '100%',
        }}>
    <CodeBlock className={styles.codeBlock} language="python">
      {children}
    </CodeBlock>
  </div>);
};
