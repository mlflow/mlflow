export function Table({ children }) {
    return (<div className="w-full overflow-x-auto">
      <table>{children}</table>
    </div>);
}
