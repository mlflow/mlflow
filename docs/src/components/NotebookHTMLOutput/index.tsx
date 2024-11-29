export const NotebookHTMLOutput = ({ children }): JSX.Element => (
  <div
    style={{
      display: "flex",
      flexDirection: "row",
      width: "100%",
    }}
  >
    <div style={{ width: "2rem", flexShrink: 0 }} />
    <div style={{ flexGrow: 1, minWidth: 0, fontSize: "0.8rem" }}>
      {children}
    </div>
  </div>
);
