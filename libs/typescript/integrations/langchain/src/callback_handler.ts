import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { Serializable } from '@langchain/core/load/serializable';
import type { Serialized } from '@langchain/core/load/serializable';
import type { BaseMessage } from '@langchain/core/messages';
import type { DocumentInterface } from '@langchain/core/documents';
import type { AgentAction, AgentFinish } from '@langchain/core/agents';
import type { LLMResult } from '@langchain/core/outputs';
import type { ChainValues } from '@langchain/core/utils/types';

import {
  SpanType,
  SpanAttributeKey,
  SpanStatusCode,
  SpanEvent,
  startSpan,
  getCurrentActiveSpan,
  type LiveSpan
} from 'mlflow-tracing';

import { parseLLMResult, parseMessage } from './utils';

interface StartSpanArgs {
  runId: string;
  parentRunId?: string;
  spanType: SpanType;
  name: string;
  inputs?: unknown;
  attributes?: Record<string, unknown>;
}

interface SpanRegistryEntry {
  span: LiveSpan;
}

/**
 * Callback handler that forwards LangChain lifecycle events to MLflow tracing.
 */
export class MlflowCallback extends BaseCallbackHandler {
  readonly name = 'mlflow_langchain';
  private readonly spans = new Map<string, SpanRegistryEntry>();

  // ---------------------------------------------------------------------------
  // LangChain lifecycle overrides
  // ---------------------------------------------------------------------------

  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.LLM,
      name: this.assignSpanName(llm, 'llm'),
      inputs: prompts,
      attributes: { [SpanAttributeKey.MESSAGE_FORMAT]: 'langchain' }
    });
  }

  async handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    extraParams?: Record<string, unknown>
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }
    this.setTokenUsage(span, output);
    span.end({ outputs: parseLLMResult(output) });
    this.spans.delete(runId);
  }

  async handleLLMError(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    extraParams?: Record<string, unknown>
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(err);
    span.end();
    this.spans.delete(runId);
  }

  async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string | undefined,
    extraParams?: Record<string, unknown> | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.CHAT_MODEL,
      name: this.assignSpanName(llm, 'chat_model'),
      inputs: messages.map((m) => m.map((msg) => parseMessage(msg))),
      attributes: {
        [SpanAttributeKey.MESSAGE_FORMAT]: 'langchain',
        ...metadata
      }
    });
  }

  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runType?: string,
    runName?: string
  ): Promise<void> {
    // If the object is Serializable, parse it
    if ('lc_serializable' in inputs) {
      inputs = inputs.toJSON().kwargs;
    }

    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.CHAIN,
      name: this.assignSpanName(chain, 'chain'),
      inputs,
      attributes: metadata
    });
  }

  async handleChainEnd(
    outputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    kwargs?: { inputs?: Record<string, unknown> }
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    // If the object is Serializable, parse it
    if ('lc_serializable' in outputs) {
      outputs = outputs.toJSON().kwargs;
    }

    span.end({ outputs: outputs });
    this.spans.delete(runId);
  }

  async handleChainError(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    kwargs?: { inputs?: Record<string, unknown> }
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(err);
    span.end();
    this.spans.delete(runId);
  }

  async handleToolStart(
    tool: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.TOOL,
      name: this.assignSpanName(tool, 'tool'),
      inputs: this.parseToolInput(input),
      attributes: metadata
    });
  }

  async handleToolEnd(
    output: any,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.end({ outputs: output });
    this.spans.delete(runId);
  }

  async handleToolError(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(err);
    span.end();
    this.spans.delete(runId);
  }

  async handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.RETRIEVER,
      name: this.assignSpanName(retriever, 'retriever'),
      inputs: query,
      attributes: metadata
    });
  }

  async handleRetrieverEnd(
    documents: DocumentInterface[],
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.end({ outputs: documents.map(this.mapDocument) });
    this.spans.delete(runId);
  }

  async handleRetrieverError(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(err);
    span.end();
    this.spans.delete(runId);
  }

  async handleAgentAction(
    action: AgentAction,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.addEvent(
      new SpanEvent({
        name: 'agent_action',
        attributes: {
          tool: action.tool,
          tool_input: this.safeStringify(action.toolInput ?? action.tool_input),
          log: action.log
        }
      })
    );
  }

  async handleAgentEnd(
    action: AgentFinish,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.addEvent(
      new SpanEvent({
        name: 'agent_finish',
        attributes: {
          return_values: this.safeStringify(action.returnValues ?? action.return_values),
          log: action.log
        }
      })
    );
  }

  async flush(): Promise<void> {
    for (const [runId, entry] of this.spans.entries()) {
      try {
        entry.span.end({ status: SpanStatusCode.OK });
      } catch (error) {
        console.debug(`Failed to flush span for run ${runId}`, error);
      } finally {
        this.spans.delete(runId);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private startSpan(args: StartSpanArgs): LiveSpan {
    const parent = this.resolveParentSpan(args.parentRunId);
    const span = startSpan({
      name: args.name,
      spanType: args.spanType,
      inputs: args.inputs,
      attributes: args.attributes,
      parent
    });

    this.spans.set(args.runId, { span });

    return span;
  }

  private getSpan(runId: string): LiveSpan | undefined {
    return this.spans.get(runId)?.span;
  }

  private resolveParentSpan(parentRunId?: string): LiveSpan | undefined {
    if (parentRunId) {
      const parentSpan = this.spans.get(parentRunId)?.span;
      if (parentSpan) {
        return parentSpan;
      }
    }

    const active = getCurrentActiveSpan();
    return active ?? undefined;
  }

  private assignSpanName(serialized: Serialized | undefined, defaultName: string): string {
    if (!serialized) {
      return defaultName;
    }

    const nameField = (serialized as any).name as string | undefined;
    if (nameField) {
      return nameField;
    }

    const idField = (serialized as any).id;
    if (Array.isArray(idField) && idField.length > 0) {
      const last = idField[idField.length - 1];
      if (typeof last === 'string') {
        return last;
      }
    }

    return defaultName;
  }

  private parseToolInput(raw: string): unknown {
    try {
      return JSON.parse(raw);
    } catch (error) {
      return raw;
    }
  }

  private mapDocument(doc: DocumentInterface): Record<string, unknown> {
    return {
      page_content: (doc as any).pageContent ?? (doc as any).page_content,
      metadata: { ...(doc as any).metadata }
    };
  }

  private setTokenUsage(span: LiveSpan, result: LLMResult): void {
    const usage = (result.llmOutput as any)?.tokenUsage || (result.llmOutput as any)?.usage;
    if (usage && usage.totalTokens) {
      const parsedUsage = {
        input_tokens: usage.promptTokens,
        output_tokens: usage.completionTokens,
        total_tokens: usage.totalTokens
      };
      span.setAttribute(SpanAttributeKey.TOKEN_USAGE, parsedUsage);
    }
  }

  private safeStringify(value: unknown): string | undefined {
    if (value == null) {
      return undefined;
    }
    try {
      return typeof value === 'string' ? value : JSON.stringify(value);
    } catch (error) {
      return String(value);
    }
  }
}
