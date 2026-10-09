const MESSAGE_PATTERNS: Array<{ pattern: RegExp; message: string }> = [
  {
    pattern: /System One models only support structured evaluation through \/gateway\/typesafe\/v1\/systemone/i,
    message:
      'This model is decision-only and cannot process normal chat requests. Choose a chat model for this request, or use the endpoint only with Boolean or categorical judges.',
  },
  {
    pattern: /System One endpoints cannot mix System One and chat models/i,
    message:
      'This endpoint mixes decision-only and chat models. Use only chat models for normal requests, or use only TypeSafe/OpenRouter Jev models for Boolean or categorical judges.',
  },
  {
    pattern: /Gateway endpoint does not use a System One model/i,
    message:
      'This endpoint does not use a TypeSafe or OpenRouter Jev decision model, so it cannot run this structured judge.',
  },
  {
    pattern: /TypeSafe judge models do not support trace-based evaluation/i,
    message:
      'This model does not support the full {{ trace }} variable. Rewrite the instructions using {{ inputs }}, {{ outputs }}, {{ expectations }}, or {{ conversation }}.',
  },
  {
    pattern: /TypeSafe judge models support bool or finite Literal feedback value types/i,
    message:
      'This model only supports Boolean or a fixed list of categorical outcomes. Change the output type, update the categorical choices, or choose a chat model.',
  },
];

export function formatJudgeModelError(error: unknown): string | null {
  const errorObject = error as { message?: string; displayMessage?: string };
  const message = (error instanceof Error && error.message) || errorObject?.message || errorObject?.displayMessage;

  if (!message) {
    return null;
  }

  return MESSAGE_PATTERNS.find(({ pattern }) => pattern.test(message))?.message ?? null;
}
