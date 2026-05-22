/**
 * OSS stub for the trace-explorer modal that opens from a record's `Source` row in the v2
 * detail page. Universe's implementation embeds a managed-evals trace viewer; OSS's trace
 * explorer lives at a different surface (Traces tab) and isn't a modal yet, so for now this
 * is a no-op. The records-side-panel link will open the modal but render nothing — wiring
 * up the real OSS trace viewer is a follow-up.
 *
 * TODO: replace with a real OSS trace viewer modal (likely wrapping ModelTraceExplorer
 * under `@databricks/web-shared/model-trace-explorer`).
 */
export interface TraceModalProps {
  visible: boolean;
  onClose: () => void;
  traceId: string;
  experimentId: string;
  selectedSqlWarehouseId?: string;
}

export const TraceModal = (_props: TraceModalProps) => null;
