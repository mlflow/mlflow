export function TreeView({ items }) {
    if (!items || items.length === 0) {
        return null;
    }
    return (<ul style={{ fontSize: '1.25rem' }}>
      {items.map((i) => (<>
          <li className="badge badge--info">{i.name}</li>
          {i.items ? <TreeView items={i.items}/> : <br />}
        </>))}
    </ul>);
}
