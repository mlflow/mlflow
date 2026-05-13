import { Span as OTelSpan, ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { LiveSpan } from '../core/entities/span';
import { InMemoryTraceManager } from '../core/trace_manager';

/**
 * Hooks to be executed by the span processor.
 * Primary used for adding custom processing to the span for autologging integrations.
 */
type OnSpanStartHook = (span: LiveSpan) => void;
type OnSpanEndHook = (span: LiveSpan) => void;

const onSpanStartHooks: Set<OnSpanStartHook> = new Set();
const onSpanEndHooks: Set<OnSpanEndHook> = new Set();

export function registerOnSpanStartHook(hook: OnSpanStartHook): void {
  onSpanStartHooks.add(hook);
}

export function getOnSpanStartHooks(): OnSpanStartHook[] {
  return Array.from(onSpanStartHooks);
}

export function registerOnSpanEndHook(hook: OnSpanEndHook): void {
  onSpanEndHooks.add(hook);
}

export function getOnSpanEndHooks(): OnSpanEndHook[] {
  return Array.from(onSpanEndHooks);
}

export function executeOnSpanStartHooks(span: OTelSpan): void {
  // Execute onStart hooks for autologging integrations
  const hooks = getOnSpanStartHooks();
  if (hooks.length === 0) {
    return;
  }

  const mlflowSpan = getMlflowSpan(span);
  if (!mlflowSpan) {
    return;
  }

  for (const hook of hooks) {
    try {
      hook(mlflowSpan);
    } catch (error) {
      console.debug('Error executing onStart hook:', error);
    }
  }
}

export function executeOnSpanEndHooks(span: OTelReadableSpan): void {
  const hooks = getOnSpanEndHooks();
  if (hooks.length === 0) {
    return;
  }

  const mlflowSpan = getMlflowSpan(span);
  if (!mlflowSpan) {
    return;
  }

  for (const hook of hooks) {
    try {
      hook(mlflowSpan);
    } catch (error) {
      console.debug('Error executing onEnd hook:', error);
    }
  }
}

function getMlflowSpan(span: OTelSpan | OTelReadableSpan): LiveSpan | null {
  const traceManager = InMemoryTraceManager.getInstance();
  const traceId = traceManager.getMlflowTraceIdFromOtelId(span.spanContext().traceId);
  return traceManager.getSpan(traceId, span.spanContext().spanId);
}
