export type TraceResponseRequest = { id: number; body: ArrayBuffer };
export type TraceResponseMessage =
  | { type: 'ready' }
  | { type: 'parsed'; id: number; value: unknown }
  | { type: 'failed'; id: number; message: string };

export const parseTraceResponseBody = (body: ArrayBuffer): unknown => JSON.parse(new TextDecoder().decode(body));
