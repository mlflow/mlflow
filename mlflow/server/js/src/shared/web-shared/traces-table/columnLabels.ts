import { defineMessages } from '@databricks/i18n';
import type { TraceColumnId } from './types';

export const TRACE_COLUMN_LABELS = defineMessages({
  trace_id: { defaultMessage: 'Trace ID', description: 'Header for the traces table trace-id column' },
  trace_name: { defaultMessage: 'Trace name', description: 'Header for the traces table trace-name column' },
  start_time: { defaultMessage: 'Time', description: 'Header for the traces table start-time column' },
  input: { defaultMessage: 'Input', description: 'Header for the traces table input column' },
  output: { defaultMessage: 'Output', description: 'Header for the traces table output column' },
  user: { defaultMessage: 'User', description: 'Header for the traces table user column' },
  session: { defaultMessage: 'Session', description: 'Header for the traces table session column' },
  duration: { defaultMessage: 'Duration', description: 'Header for the traces table duration column' },
  state: { defaultMessage: 'State', description: 'Header for the traces table state column' },
  source: { defaultMessage: 'Source', description: 'Header for the traces table source column' },
  run_name: { defaultMessage: 'Run name', description: 'Header for the traces table run-name column' },
  tokens: { defaultMessage: 'Tokens', description: 'Header for the traces table tokens column' },
  cost: { defaultMessage: 'Cost', description: 'Header for the traces table cost column' },
  tags: { defaultMessage: 'Tags', description: 'Header for the traces table tags column' },
} as const);
