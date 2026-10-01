/**
 * MLflow Tracing wrapper for the TypeSafe AI JavaScript SDK.
 */

import {
  getCurrentActiveSpan,
  SpanAttributeKey,
  SpanStatusCode,
  SpanType,
  startSpan,
  type LiveSpan,
  type TokenUsage,
} from '@mlflow/core';
import type { TypeSafeClient } from '@typesafe-ai/sdk';

const SPAN_NAME = 'typesafe.system_one';
const MODEL_ATTRIBUTE = 'mlflow.llm.model';
const MODEL_PROVIDER_ATTRIBUTE = 'mlflow.llm.provider';
const REQUEST_ID_ATTRIBUTE = 'typesafe.request_id';
const REQUEST_ID_HEADER = 'x-typesafe-request-id';

type SystemOneMethod = TypeSafeClient['systemOne'];
type SystemOneParameters = Parameters<SystemOneMethod>;
type SystemOnePromise = ReturnType<SystemOneMethod>;

/**
 * Return a traced TypeSafe client that records each `systemOne` request as an MLflow span.
 *
 * The wrapper preserves the TypeSafe SDK's `APIPromise`, including `asResponse()`,
 * `withResponse()`, and `map()`.
 */
export function tracedTypeSafe<T extends TypeSafeClient>(client: T): T {
  return new Proxy(client, {
    get(target, prop) {
      if (prop === 'systemOne') {
        return (...args: SystemOneParameters): SystemOnePromise => traceSystemOne(target, args);
      }

      // TypeSafeClient uses JavaScript private fields. Bind methods to the real
      // instance so their `this` value is never the proxy.
      const value = Reflect.get(target, prop, target) as unknown;
      return typeof value === 'function'
        ? (value as (...args: unknown[]) => unknown).bind(target)
        : value;
    },
  });
}

function traceSystemOne(client: TypeSafeClient, args: SystemOneParameters): SystemOnePromise {
  const [request] = args;
  const effectiveModel = request.model ?? client.defaultModel;
  const inputs = serializeLikeRequestBody({ ...request, model: effectiveModel });
  const parent = getCurrentActiveSpan();
  const span = startSpan({
    name: SPAN_NAME,
    spanType: SpanType.LLM,
    parent: parent ?? undefined,
  });

  span.setInputs(inputs);
  span.setAttribute(MODEL_PROVIDER_ATTRIBUTE, 'typesafe');
  if (effectiveModel) {
    span.setAttribute(MODEL_ATTRIBUTE, effectiveModel);
  }

  let apiPromise: SystemOnePromise;
  try {
    apiPromise = client.systemOne(...args);
  } catch (error) {
    endSpanWithError(span, error);
    throw error;
  }

  // TypeSafe returns an APIPromise with response helpers. Awaiting it here or
  // wrapping it in withSpan() would replace it with a native Promise and break
  // those helpers. Observe the shared response in a sidecar instead, cloning its
  // body so callers can still consume either the parsed or raw response.
  try {
    void apiPromise
      .asResponse()
      .then(
        async (response) => {
          try {
            const output = await parseResponseClone(response);
            if (output !== undefined) {
              span.setOutputs(output);
            }
            setResponseAttributes(span, output, effectiveModel);
            setRequestId(span, response.headers.get(REQUEST_ID_HEADER));
          } catch (error) {
            // Tracing must never change a successful TypeSafe request into a user
            // error. End the span with whatever data was captured successfully.
            console.debug('Failed to capture the TypeSafe response for MLflow tracing', error);
          } finally {
            span.end();
          }
        },
        (error: unknown) => {
          setRequestId(span, requestIdFromError(error));
          endSpanWithError(span, error);
        },
      )
      .catch((error: unknown) => {
        // This detached observer must never create an unhandled rejection.
        console.debug('Failed to finish the TypeSafe trace observer', error);
        span.end();
      });
  } catch (error) {
    // A conforming TypeSafe APIPromise does not throw from asResponse(), but
    // keep instrumentation best-effort if a custom client violates that contract.
    console.debug('Failed to observe the TypeSafe response for MLflow tracing', error);
    span.end();
  }

  return apiPromise;
}

function serializeLikeRequestBody(value: unknown): unknown {
  try {
    // TypeSafe sends requests with JSON.stringify. Round-tripping here removes
    // optional `undefined` fields from question builders just like the wire body.
    return JSON.parse(JSON.stringify(value)) as unknown;
  } catch {
    // Let the SDK remain responsible for rejecting invalid request values.
    return value;
  }
}

async function parseResponseClone(response: Response): Promise<unknown> {
  const text = await response.clone().text();
  if (text.length === 0) {
    return undefined;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function setResponseAttributes(span: LiveSpan, output: unknown, fallbackModel: string): void {
  if (!isRecord(output)) {
    return;
  }

  const model = typeof output.model === 'string' && output.model ? output.model : fallbackModel;
  if (model) {
    span.setAttribute(MODEL_ATTRIBUTE, model);
  }

  const usage = parseTokenUsage(output.usage);
  if (usage) {
    span.setAttribute(SpanAttributeKey.TOKEN_USAGE, usage);
  }
}

function parseTokenUsage(value: unknown): TokenUsage | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const inputTokens = value.input_tokens;
  const outputTokens = value.output_tokens;
  if (
    typeof inputTokens !== 'number' ||
    !Number.isFinite(inputTokens) ||
    typeof outputTokens !== 'number' ||
    !Number.isFinite(outputTokens)
  ) {
    return undefined;
  }

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
  };
}

function endSpanWithError(span: LiveSpan, value: unknown): void {
  const error = value instanceof Error ? value : new Error(String(value));
  span.setStatus(SpanStatusCode.ERROR, `${error.name}: ${error.message}`);
  span.recordException(error);
  span.end();
}

function requestIdFromError(value: unknown): string | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  return typeof value.requestId === 'string' ? value.requestId : undefined;
}

function setRequestId(span: LiveSpan, requestId: string | null | undefined): void {
  if (requestId) {
    span.setAttribute(REQUEST_ID_ATTRIBUTE, requestId);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null;
}
