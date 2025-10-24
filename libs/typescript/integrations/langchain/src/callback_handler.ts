import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Serialized } from '@langchain/core/load/serializable';
import type { BaseMessage } from '@langchain/core/messages';
import type { DocumentInterface } from '@langchain/core/documents';
import type { AgentAction, AgentFinish } from '@langchain/core/agents';
import type { ChatGenerationChunk, GenerationChunk, LLMResult } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ChainValues } from "@langchain/core/utils/types";

import {
  SpanType,
  SpanAttributeKey,
  SpanStatusCode,
  SpanEvent,
  startSpan,
  getCurrentActiveSpan,
  type LiveSpan
} from 'mlflow-tracing';

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
    serialized: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    _extra?: unknown
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.LLM,
      name: this.assignSpanName(serialized, 'llm'),
      inputs: prompts,
      attributes: { [SpanAttributeKey.MESSAGE_FORMAT]: 'langchain' }
    });
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }
    this.setTokenUsage(span, output);
    span.end({ outputs: output });
    this.spans.delete(runId);
  }

  async handleLLMError(error: Error, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(error);
    span.end();
    this.spans.delete(runId);
  }

  async handleChatModelStart(
    serialized: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    config?: RunnableConfig
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.CHAT_MODEL,
      name: this.assignSpanName(serialized, 'chat_model'),
      inputs: messages,
      attributes: {
        [SpanAttributeKey.MESSAGE_FORMAT]: 'langchain',
        ...this.buildAttributes({ config })
      }
    });
  }

  async handleChatModelError(error: Error, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(error);
    span.end();
    this.spans.delete(runId);
  }

  async handleChainStart(
    serialized: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    config?: RunnableConfig
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.CHAIN,
      name: this.assignSpanName(serialized, 'chain'),
      inputs,
      attributes: this.buildAttributes({ config })
    });
  }

  async handleChainEnd(outputs: ChainValues, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.end({ outputs: outputs });
    this.spans.delete(runId);
  }

  async handleChainError(error: Error, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(error);
    span.end();
    this.spans.delete(runId);
  }

  async handleToolStart(
    serialized: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    kwargs?: Record<string, unknown>
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.TOOL,
      name: this.assignSpanName(serialized, 'tool'),
      inputs: this.parseToolInput(input),
      attributes: this.buildAttributes({ kwargs })
    });
  }

  async handleToolEnd(output: unknown, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.end({ outputs: output });
    this.spans.delete(runId);
  }

  async handleToolError(error: Error, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(error);
    span.end();
    this.spans.delete(runId);
  }

  async handleRetrieverStart(
    serialized: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    config?: RunnableConfig
  ): Promise<void> {
    this.startSpan({
      runId,
      parentRunId,
      spanType: SpanType.RETRIEVER,
      name: this.assignSpanName(serialized, 'retriever'),
      inputs: query,
      attributes: this.buildAttributes({ config })
    });
  }

  async handleRetrieverEnd(documents: DocumentInterface[], runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.end({ outputs: documents.map(this.mapDocument) });
    this.spans.delete(runId);
  }

  async handleRetrieverError(error: Error, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.recordException(error);
    span.end();
    this.spans.delete(runId);
  }

  async handleAgentAction(action: AgentAction, runId: string): Promise<void> {
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

  async handleAgentEnd(action: AgentFinish, runId: string): Promise<void> {
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

  async handleRetry(data: { attempt: number; maxAttempts?: number; error?: Error }, runId: string): Promise<void> {
    const span = this.getSpan(runId);
    if (!span) {
      return;
    }

    span.addEvent(
      new SpanEvent({
        name: 'retry',
        attributes: {
          attempt: data.attempt,
          max_attempts: data.maxAttempts,
          error: data.error ? data.error.message : undefined
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

  private buildAttributes(input: {
    config?: RunnableConfig;
    kwargs?: Record<string, unknown>;
  }): Record<string, unknown> | undefined {
    const attributes: Record<string, unknown> = {};
    if (input.config?.metadata) {
      attributes.metadata = input.config.metadata;
    }
    if (input.kwargs) {
      attributes.invocation_params = input.kwargs;
    }
    if (Object.keys(attributes).length === 0) {
      return undefined;
    }
    return attributes;
  }

  private setTokenUsage(span: LiveSpan, result: LLMResult): void {
    const usage = (result.llmOutput as any)?.tokenUsage || (result.llmOutput as any)?.usage;
    if (usage && usage.totalTokens) {
      const parsedUsage = {
        input_tokens: usage.promptTokens,
        output_tokens: usage.completionTokens,
        total_tokens: usage.totalTokens,
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
