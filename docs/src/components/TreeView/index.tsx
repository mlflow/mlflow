type TreeViewItem = {
  name: string;
  items?: TreeViewItem[];
};

type TreeViewProps = {
  items?: TreeViewItem[];
};

export function TreeView({ items }: TreeViewProps) {
  if (!items || items.length === 0) {
    return null;
  }

  return (
    <ul style={{ fontSize: '1.25rem' }}>
      {items.map((i) => (
        <>
          <li className="badge badge--info">{i.name}</li>
          {i.items ? <TreeView items={i.items} /> : <br />}
        </>
      ))}
    </ul>
  );
}
