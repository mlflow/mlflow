import { compact, has, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

export type GeminiChatInput = {
  contents: string | GeminiContent[];
};

export type GeminiChatOutput = {
  candidates: GeminiCandidate[];
  // propmtFeedback: GeminiPropmptFeedback;
  // usageMetadata: GeminiUsageMetadata;
  modelVersion: string;
  responseId: string;
};

type GeminiCandidate = {
  content: GeminiContent;
  finishReason:
    | 'FINISH_REASON_UNSPECIFIED'
    | 'STOP'
    | 'MAX_TOKENS'
    | 'SAFETY'
    | 'RECITATION'
    | 'LANGUAGE'
    | 'OTHER'
    | 'BLOCKLIST'
    | 'PROHIBITED_CONTENT'
    | 'SPII'
    | 'MALFORMED_FUNCTION_CAL'
    | 'IMAGE_SAFETY'
    | 'UNEXPECTED_TOOL_CAL';
  // safetyRatings: GeminiSafetyRating[]
  // citationMetadata: GeminiCitationMetadata
  // tokenCount: number
  // groundingAttributions: GeminiGroundingAttribution[]
  // groundingMetadata: GeminiGroundingMetadata
  // avgLogprobs: number
  // logprobsResult: GeminiLogprobsResult
  // urlContextMetadata: GeminiUrlContextMetadata
  // index: number
};

type GeminiContent = {
  role: 'user' | 'model';
  parts: GeminiContentPart[];
};

type GeminiContentPart = { text: string };
// | { inlineData: GeminiBlob }
// | { functionCall: GeminiFunctionCall }
// | { functionResponse: GeminiFunctionResponse }
// | { fileData: GeminiFileData }
// | { executableCode: GeminiExecutableCode }
// | { codeExecutionResult: GeminiCodeExecutionResult };

// type GeminiBlob = {
//   mimeType: string;
//   data: string;
// };

// type GeminiFunctionCall = {
//   id: string;
//   name: string;
//   args: Record<string, string>;
// };

// type GeminiFunctionResponse = {
//   id: string;
//   name: string;
//   response: Record<string, string>;
//   willContinue: boolean;
//   scheduling: 'SCHEDULING_UNSPECIFIED' | 'SILENT' | 'WHEN_IDLE' | 'INTERRUPT';
// };

// type GeminiFileData = {
//   mimeType: string;
//   fileUri: string;
// };

// type GeminiExecutableCode = {
//   language: 'LANGUAGE_UNSPECIFIED' | 'PYTHON';
//   code: string;
// };

// type GeminiCodeExecutionResult = {
//   outcome: 'OUTCOME_UNSPECIFIED' | 'OUTCOME_OK' | 'OUTCOME_FAILED' | 'OUTCOME_DEADLINE_EXCEEDED';
//   output: string;
// };

const isGeminiContentPart = (obj: unknown): obj is GeminiContentPart => {
  return isObject(obj) && 'text' in obj && isString(obj.text);
};

const isGeminiContent = (obj: unknown): obj is GeminiContent => {
  return (
    isObject(obj) &&
    'role' in obj &&
    isString(obj.role) &&
    ['user', 'model'].includes(obj.role) &&
    has(obj, 'parts') &&
    Array.isArray(obj.parts) &&
    obj.parts.every(isGeminiContentPart)
  );
};

const isGeminiCandidate = (obj: unknown): obj is GeminiCandidate => {
  return isObject(obj) && 'content' in obj && isGeminiContent(obj.content);
};

export const normalizeGeminiChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('contents' in obj) {
    if (isString(obj.contents)) {
      const message = prettyPrintChatMessage({ type: 'message', content: obj.contents, role: 'user' });
      return message && [message];
    }

    if (isArray(obj.contents) && obj.contents.every(isGeminiContent)) {
      return compact(
        obj.contents.map((item) => {
          const role = item.role === 'model' ? 'assistant' : item.role;
          return prettyPrintChatMessage({
            type: 'message',
            content: item.parts.map((part) => ({ type: 'text', text: part.text })),
            role,
          });
        }),
      );
    }
  }

  return null;
};

export const normalizeGeminiChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isObject(obj)) {
    return null;
  }

  if ('candidates' in obj && isArray(obj.candidates) && obj.candidates.every(isGeminiCandidate)) {
    return compact(
      obj.candidates
        .flatMap((item) => item.content)
        .map((item) => {
          const role = item.role === 'model' ? 'assistant' : item.role;
          return prettyPrintChatMessage({
            type: 'message',
            content: item.parts.map((part) => ({ type: 'text', text: part.text })),
            role,
          });
        }),
    );
  }

  return null;
};
