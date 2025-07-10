export type ChatCompletionUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: {
    cached_tokens: number;
    audio_tokens: number;
  };
  completion_tokens_details?: {
    reasoning_tokens: number;
    audio_tokens: number;
    accepted_prediction_tokens: number;
    rejected_prediction_tokens: number;
  };
};

export type ResponsesUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
};

export type OpenAIUsage = ChatCompletionUsage | ResponsesUsage;
