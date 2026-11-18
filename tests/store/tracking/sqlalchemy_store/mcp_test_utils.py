from mlflow.store.tracking.dbmodels.models import SqlSpanMCPServerLink


def get_mcp_server_links_for_test(store, trace_id: str) -> list[tuple[str, str, str | None]]:
    with store.ManagedSessionMaker() as session:
        rows = (
            session
            .query(
                SqlSpanMCPServerLink.mcp_workspace,
                SqlSpanMCPServerLink.mcp_name,
                SqlSpanMCPServerLink.mcp_version,
            )
            .filter(SqlSpanMCPServerLink.trace_id == trace_id)
            .all()
        )
    # SQL DISTINCT follows database collation and can merge different version strings.
    return list(dict.fromkeys((workspace, name, version) for workspace, name, version in rows))
