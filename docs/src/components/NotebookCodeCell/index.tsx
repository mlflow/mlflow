import CodeBlock from '@theme/CodeBlock';
import styles from './styles.module.css';

export const NotebookCodeCell = ({ children, executionCount }): JSX.Element => (
  <div
    style={{
      flexGrow: 1,
      minWidth: 0,
      marginTop: 'var(--padding-md)',
      width: '100%',
    }}
  >
    <CodeBlock className={styles.codeBlock} language="python">
      {children}
    </CodeBlock>
  </div>
);
