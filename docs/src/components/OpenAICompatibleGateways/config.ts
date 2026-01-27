export interface OpenAICompatibleGateway {
  id: string;
  name: string;
  displayName?: string; // e.g., "Kong AI Gateway" - if different from name
  description: string; // Intro paragraph about the gateway
  baseUrl: string;
  apiKeyPlaceholder: string;
  sampleModel: string;
  heroImage?: string; // Optional custom hero image path
  defaultHeaders?: {
    // For gateways requiring custom headers (e.g., Helicone, Portkey)
    python: Record<string, string>;
    typescript: Record<string, string>;
  };
  headerComment?: string; // Comment explaining header usage
  prerequisite: string; // HTML content for prerequisite info block
}

export const OPENAI_COMPATIBLE_GATEWAYS: OpenAICompatibleGateway[] = [
  {
    id: 'openrouter',
    name: 'OpenRouter',
    description:
      '<a href="https://openrouter.ai/">OpenRouter</a> is a unified API gateway that provides access to 280+ LLMs from providers like OpenAI, Anthropic, Google, Meta, and many others through a single OpenAI-compatible API. This allows developers to easily switch between models without changing their code.',
    baseUrl: 'https://openrouter.ai/api/v1',
    apiKeyPlaceholder: '<YOUR_OPENROUTER_API_KEY>',
    sampleModel: 'anthropic/claude-sonnet-4.5',
    heroImage: '/images/llms/openrouter/openrouter-tracing.png',
    prerequisite:
      'Before following the steps below, you need to create an <ins><a href="https://openrouter.ai/">OpenRouter account</a></ins> and generate an API key from the <ins><a href="https://openrouter.ai/keys">Keys page</a></ins>.',
  },
  {
    id: 'vercel-ai-gateway',
    name: 'Vercel AI Gateway',
    description:
      '<a href="https://vercel.com/docs/ai-gateway">Vercel AI Gateway</a> provides a unified API to access hundreds of LLMs through a single endpoint. Key features include high reliability with automatic fallbacks to other providers, spend monitoring across providers, and zero markup on token costs. It works seamlessly with the OpenAI SDK, Anthropic SDK, and Vercel AI SDK.',
    baseUrl: 'https://ai-gateway.vercel.sh/v1',
    apiKeyPlaceholder: '<YOUR_VERCEL_AI_GATEWAY_API_KEY>',
    sampleModel: 'anthropic/claude-sonnet-4.5',
    prerequisite:
      'Create a <ins><a href="https://vercel.com/">Vercel account</a></ins> and enable <ins><a href="https://vercel.com/docs/ai-gateway">AI Gateway</a></ins> for your project. You can find your API key in the project settings.',
  },
  {
    id: 'truefoundry',
    name: 'TrueFoundry',
    displayName: 'TrueFoundry AI Gateway',
    description:
      '<a href="https://www.truefoundry.com/ai-gateway">TrueFoundry AI Gateway</a> is an enterprise-grade LLM gateway that provides access to 1000+ LLMs through a unified OpenAI-compatible API. It offers built-in governance, observability, rate limiting, and cost controls for production AI applications.',
    baseUrl: 'https://<your-control-plane>.truefoundry.cloud/api/llm/v1',
    apiKeyPlaceholder: '<YOUR_TRUEFOUNDRY_API_KEY>',
    sampleModel: 'openai/gpt-4o',
    prerequisite:
      'Create a <ins><a href="https://www.truefoundry.com/">TrueFoundry account</a></ins> with at least one model provider configured, then generate an API key from the TrueFoundry dashboard.',
  },
  {
    id: 'kong',
    name: 'Kong AI Gateway',
    description:
      '<a href="https://konghq.com/products/kong-ai-gateway">Kong AI Gateway</a> is an enterprise-grade API gateway that provides a unified OpenAI-compatible API to access multiple LLM providers including OpenAI, Anthropic, Azure, AWS Bedrock, Google Gemini, and more. It offers built-in rate limiting, caching, load balancing, and observability.',
    baseUrl: 'http://<your-kong-gateway>:8000/v1',
    apiKeyPlaceholder: '<YOUR_API_KEY>',
    sampleModel: 'gpt-4o',
    prerequisite:
      'Set up <ins><a href="https://konghq.com/products/kong-ai-gateway">Kong AI Gateway</a></ins> by following the installation guide and configure your LLM provider credentials.',
  },
  {
    id: 'helicone',
    name: 'Helicone',
    displayName: 'Helicone AI Gateway',
    description:
      '<a href="https://www.helicone.ai/">Helicone AI Gateway</a> is an open-source LLM gateway that provides unified access to 100+ AI models through an OpenAI-compatible API. It offers built-in caching, rate limiting, automatic failover, and comprehensive analytics with minimal latency overhead.',
    baseUrl: 'http://localhost:8080/ai',
    apiKeyPlaceholder: 'placeholder-api-key',
    sampleModel: 'anthropic/claude-4-5-sonnet',
    prerequisite:
      'Before following the steps below, you need to set up Helicone AI Gateway server.<ol><li>Set up your <code>.env</code> file with your LLM provider API keys (e.g., <code>OPENAI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>).</li><li>Run the gateway locally with <code>npx @helicone/ai-gateway@latest</code>.</li></ol>See the <ins><a href="https://docs.helicone.ai/gateway/overview#ai-gateway-overview">Helicone AI Gateway docs</a></ins> for more details.',
  },
  {
    id: 'portkey',
    name: 'Portkey',
    description:
      '<a href="https://portkey.ai/">Portkey</a> is an enterprise-grade AI gateway that provides unified access to 1600+ LLMs through a single OpenAI-compatible API. It offers built-in guardrails, observability, caching, load balancing, and fallback mechanisms for production AI applications.',
    baseUrl: 'https://api.portkey.ai/v1',
    apiKeyPlaceholder: '<YOUR_PORTKEY_API_KEY>',
    sampleModel: 'gpt-4o',
    defaultHeaders: {
      python: { 'x-portkey-provider': 'openai' },
      typescript: { 'x-portkey-provider': 'openai' },
    },
    headerComment: 'or "anthropic", "google", etc.',
    prerequisite:
      'Create a <ins><a href="https://portkey.ai/">Portkey account</a></ins> and generate an API key from the <ins><a href="https://app.portkey.ai/api-keys">API Keys page</a></ins>. Configure your virtual keys for the LLM providers you want to use.',
  },
];

export function getGateway(id: string): OpenAICompatibleGateway | undefined {
  return OPENAI_COMPATIBLE_GATEWAYS.find((g) => g.id === id);
}
