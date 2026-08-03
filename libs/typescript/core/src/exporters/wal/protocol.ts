/**
 * Wire protocol for the daemon's IPC socket.
 *
 * op: 'append' resolves only after the record is durable in
 * queue.log (the daemon's batching writer fsyncs the batch before
 * acknowledging).
 */

import { WalRecord } from './types';

export interface AppendRequest {
  op: 'append';
  record: WalRecord;
}

export type IpcRequest = AppendRequest;

export interface OkResponse {
  ok: true;
}

export interface ErrResponse {
  ok: false;
  error: string;
}

export type IpcResponse = OkResponse | ErrResponse;
