import { TraceData } from '../../core/entities/trace_data';
import { TraceInfo } from '../../core/entities/trace_info';

export interface ArtifactsClient {
  uploadTraceData(traceInfo: TraceInfo, traceData: TraceData): Promise<void>;
  downloadTraceData(traceInfo: TraceInfo): Promise<TraceData>;
}
