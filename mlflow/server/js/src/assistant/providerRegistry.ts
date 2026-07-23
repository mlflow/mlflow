/**
 * Shared registry of assistant providers: the single source of truth for a provider id's
 * human display name and brand logo. Used by the setup wizard (to pick one) and the composer
 * (to show, read-only, which one is active). Keyed by the same provider ids the `/config`
 * payload uses (see `GATEWAY_PROVIDER_ID` and the server-side provider names).
 */
import AnthropicLogo from '@mlflow/mlflow/src/common/static/logos/anthropic.svg';
import OpenAiLogo from '@mlflow/mlflow/src/common/static/logos/openai.svg';
import MLflowGatewayLogo from '@mlflow/mlflow/src/common/static/logos/mlflow-gateway.svg';
import OllamaLogo from '@mlflow/mlflow/src/common/static/logos/ollama.png';

export interface AssistantProvider {
  id: string;
  name: string;
  /** Longer copy for the setup card; the composer only uses `name` + `logo`. */
  description: string;
  logo: string;
  available: boolean;
}

export const ASSISTANT_PROVIDERS: AssistantProvider[] = [
  {
    id: 'claude_code',
    name: 'Claude Code',
    description: "AI assistant powered by Anthropic's Claude. Requires Claude Code CLI installed locally.",
    logo: AnthropicLogo,
    available: true,
  },
  {
    id: 'mlflow_gateway',
    name: 'MLflow AI Gateway',
    description: 'AI assistant backed by an MLflow AI Gateway deployment. Routes to any configured chat endpoint.',
    logo: MLflowGatewayLogo,
    available: true,
  },
  {
    id: 'ollama',
    name: 'Ollama',
    description: 'AI assistant using a locally running Ollama server. Requires Ollama installed and running.',
    logo: OllamaLogo,
    available: true,
  },
  {
    id: 'codex',
    name: 'OpenAI Codex',
    description:
      'AI assistant powered by OpenAI via the Codex CLI. Requires the codex CLI to be installed and authenticated.',
    logo: OpenAiLogo,
    available: true,
  },
];

/** Look up a provider's display metadata by id; undefined for unknown ids. */
export const getAssistantProvider = (id: string): AssistantProvider | undefined =>
  ASSISTANT_PROVIDERS.find((provider) => provider.id === id);
