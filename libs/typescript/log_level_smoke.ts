/**
 * End-to-end smoke for the span log-level feature, driven from the TypeScript SDK.
 *
 * Mirror of /tmp/log_level_smoke.py for users who'd rather drive from JS/TS.
 *
 * Setup (one-time, run from repo root):
 *   1. cd libs/typescript && npm install
 *
 * Run (from repo root, with the dev server up):
 *   cd libs/typescript && npx tsx log_level_smoke.ts
 *
 * The script must live inside libs/typescript/ so Node resolves @mlflow/core via
 * the workspace's node_modules — module resolution walks up from the script's
 * directory, not the cwd, so /tmp won't find it.
 *
 * Prereq: dev server running. From repo root:
 *   nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &
 */

import * as mlflow from '@mlflow/core';
import {
  SpanLogLevel,
  SpanType,
  defaultLogLevelForSpanType,
} from '@mlflow/core';

async function main() {
  await mlflow.init({
    trackingUri: 'http://localhost:5000',
    experimentId: '3',
  });

  // -------------------------------------------------------------------------
  // Trace A: nested spans with explicit levels at every node, exercising the
  // three input forms accepted by the kwarg (string / int / enum).
  // -------------------------------------------------------------------------
  const traceA = await mlflow.withSpan(
    async (root) => {
      await mlflow.withSpan(
        async (middle) => {
          await mlflow.withSpan(
            async () => {},
            {
              name: 'A_leaf_debug',
              spanType: SpanType.PARSER,
              parent: middle,
              logLevel: SpanLogLevel.DEBUG,
            },
          );
          await mlflow.withSpan(
            async () => {},
            {
              name: 'A_leaf_warning',
              spanType: SpanType.TOOL,
              parent: middle,
              logLevel: SpanLogLevel.WARNING,
            },
          );
        },
        {
          name: 'A_middle_info',
          spanType: SpanType.CHAIN,
          parent: root,
          logLevel: 20, // int form (== INFO)
        },
      );
      return mlflow.getLastActiveTraceId();
    },
    {
      name: 'A_root_debug',
      spanType: SpanType.AGENT,
      logLevel: 'DEBUG',
    },
  );

  // -------------------------------------------------------------------------
  // Trace B: autolog-style. Every span gets stamped with the helper's
  // default for its span type -- exactly what the integration packages
  // (openai, anthropic, gemini, ...) now do internally.
  // -------------------------------------------------------------------------
  const traceB = await mlflow.withSpan(
    async (root) => {
      for (const [name, type] of [
        ['B_openai_chat', SpanType.CHAT_MODEL],
        ['B_search_tool', SpanType.TOOL],
        ['B_output_parser', SpanType.PARSER],
      ] as const) {
        await mlflow.withSpan(
          async () => {},
          { name, spanType: type, parent: root, logLevel: defaultLogLevelForSpanType(type) },
        );
      }
      return mlflow.getLastActiveTraceId();
    },
    {
      name: 'B_my_chain',
      spanType: SpanType.CHAIN,
      logLevel: defaultLogLevelForSpanType(SpanType.CHAIN),
    },
  );

  // -------------------------------------------------------------------------
  // Trace C: no logLevel kwarg anywhere. Spans have no mlflow.spanLogLevel
  // attribute. UI should treat as DEBUG so old traces stay visible.
  // -------------------------------------------------------------------------
  const traceC = await mlflow.withSpan(
    async (root) => {
      await mlflow.withSpan(
        async () => {},
        { name: 'C_child_unset', spanType: SpanType.LLM, parent: root },
      );
      return mlflow.getLastActiveTraceId();
    },
    { name: 'C_root_unset', spanType: SpanType.LLM },
  );

  // -------------------------------------------------------------------------
  // Trace D: a DEBUG span that throws. With "Show exceptions" toggled on, it
  // should remain visible even at WARNING+.
  // -------------------------------------------------------------------------
  const traceD = await mlflow.withSpan(
    async (root) => {
      try {
        await mlflow.withSpan(
          async () => {
            throw new Error('intentional');
          },
          {
            name: 'D_failing_debug_span',
            spanType: SpanType.TOOL,
            parent: root,
            logLevel: SpanLogLevel.DEBUG,
          },
        );
      } catch {
        /* swallow */
      }
      return mlflow.getLastActiveTraceId();
    },
    { name: 'D_root_info', spanType: SpanType.AGENT, logLevel: SpanLogLevel.INFO },
  );

  await mlflow.flushTraces();

  console.log('trace IDs:');
  console.log(`  A (manual, mixed levels)        ${traceA}`);
  console.log(`  B (autolog defaults pattern)    ${traceB}`);
  console.log(`  C (no level set)                ${traceC}`);
  console.log(`  D (debug span with exception)   ${traceD}`);
  console.log('');
  console.log("Open http://localhost:3000, pick the 'log-level-smoke-ts' experiment, then verify:");
  console.log('');
  console.log('Trace A (4 spans):');
  console.log('  threshold DEBUG     -> all 4 visible');
  console.log('  threshold INFO      -> A_leaf_debug hidden');
  console.log('  threshold WARNING   -> only A_root_warn (or its descendants under "Show parents")');
  console.log('');
  console.log('Trace B (4 spans, autolog defaults pattern):');
  console.log('  threshold INFO      -> chain + parser hidden, chat + tool visible');
  console.log('');
  console.log('Trace C (2 spans, no level):');
  console.log('  threshold DEBUG     -> visible (treated as DEBUG -- backwards-compat default)');
  console.log('  threshold INFO      -> hidden');
  console.log('');
  console.log('Trace D (debug span that threw):');
  console.log('  threshold WARNING + Show exceptions on  -> D_failing_debug_span shown');
  console.log('  threshold WARNING + Show exceptions off -> D_failing_debug_span hidden');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
