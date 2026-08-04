import type { ProviderModel } from '../types';

export function getModelCapabilities(model: ProviderModel | undefined): string[] {
  if (!model) return [];
  const caps: string[] = [];
  if (model.supports_function_calling) caps.push('Tools');
  if (model.supports_reasoning) caps.push('Reasoning');
  if (model.supports_prompt_caching) caps.push('Caching');
  if (model.supports_response_schema) caps.push('Structured');
  return caps;
}
