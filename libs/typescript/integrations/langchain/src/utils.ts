import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';

export function parseLLMResult(result: LLMResult): any {
  const res = {
    generations: result.generations.map((gen) =>
      gen
        .filter((g) => g !== undefined)
        .map((g) => {
          return {
            message: parseMessage(g.message),
            text: g.text,
            generationInfo: g.generationInfo ?? {}
          };
        })
    ),
    llmOutput: result.llmOutput
  };
  return res;
}

export function parseMessage(messages: BaseMessage): any {
  const d = messages.toDict();
  return { type: d.type, ...d.data };
}
