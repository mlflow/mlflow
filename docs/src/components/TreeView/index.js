export function TreeView(_a) {
    var items = _a.items;
    if (!items || items.length === 0) {
        return null;
    }
    return (<ul style={{ fontSize: '1.25rem' }}>
      {items.map(function (i) { return (<>
          <li className="badge badge--info">{i.name}</li>
          {i.items ? <TreeView items={i.items}/> : <br />}
        </>); })}
    </ul>);
}
