import { init } from './core/config';
import { getLastActiveTraceId, startSpan, withSpan } from './core/api';
import { flushTraces } from './core/provider';

export { getLastActiveTraceId, flushTraces, init, startSpan, withSpan };
