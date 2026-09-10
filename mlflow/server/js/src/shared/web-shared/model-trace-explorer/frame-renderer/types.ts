import type { ModelTrace } from '../ModelTrace.types';

export enum ModelTraceChildToParentFrameMessage {
  Ready = 'READY',
  LogError = 'LOG_ERROR',
  LogEvent = 'LOG_EVENT',
}

type ModelTraceFrameReadyMessage = {
  type: ModelTraceChildToParentFrameMessage.Ready;
};

type ModelTraceFrameLogErrorMessage = {
  type: ModelTraceChildToParentFrameMessage.LogError;
  error: Error | any;
  isPromiseRejection?: boolean;
};

type ModelTraceFrameLogEventMessage = {
  type: ModelTraceChildToParentFrameMessage.LogEvent;
  payload: any;
};

export enum ModelTraceParentToChildFrameMessage {
  UpdateTrace = 'UPDATE_TRACE',
}

type ModelTraceFrameUpdateTraceMessage = {
  type: ModelTraceParentToChildFrameMessage.UpdateTrace;
  traceData: ModelTrace;
};

export type ModelTraceChildToParentFrameMessageType =
  | ModelTraceFrameReadyMessage
  | ModelTraceFrameLogErrorMessage
  | ModelTraceFrameLogEventMessage;
export type ModelTraceParentToChildFrameMessageType = ModelTraceFrameUpdateTraceMessage;
