export const NotebookCellOutput = ({ children, isStderr }) => (<pre style={{
        margin: 0,
        borderRadius: 0,
        background: 'none',
        fontSize: '0.85rem',
        flexGrow: 1,
        padding: `var(--padding-sm)`,
    }}>
    {children}
  </pre>);
