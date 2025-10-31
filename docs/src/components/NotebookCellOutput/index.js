export var NotebookCellOutput = function (_a) {
    var children = _a.children, isStderr = _a.isStderr;
    return (<pre style={{
            margin: 0,
            borderRadius: 0,
            background: 'none',
            fontSize: '0.85rem',
            flexGrow: 1,
            padding: "var(--padding-sm)",
        }}>
    {children}
  </pre>);
};
