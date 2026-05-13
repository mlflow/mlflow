import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { Button, DesignSystemProvider, Typography } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { SelectorModal } from './SelectorModal';
import type { SelectorItem } from './types';

const meta: Meta<typeof SelectorModal> = {
  title: 'Common/SelectorModal',
  component: SelectorModal,
  decorators: [
    (Story) => (
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <Story />
        </IntlProvider>
      </DesignSystemProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof SelectorModal>;

const litellmProviders: SelectorItem[] = [
  { key: 'ai21', label: 'AI21', value: 'ai21', description: 'AI21 Labs language models' },
  { key: 'aleph_alpha', label: 'Aleph Alpha', value: 'aleph_alpha', description: 'European AI models' },
  { key: 'anyscale', label: 'Anyscale', value: 'anyscale', description: 'Open source model hosting' },
  { key: 'cloudflare', label: 'Cloudflare', value: 'cloudflare', description: 'Cloudflare Workers AI' },
  { key: 'deepinfra', label: 'DeepInfra', value: 'deepinfra', description: 'Inference API platform' },
  { key: 'deepseek', label: 'DeepSeek', value: 'deepseek', description: 'DeepSeek AI models' },
  { key: 'fireworks_ai', label: 'Fireworks AI', value: 'fireworks_ai', description: 'Fast inference platform' },
  { key: 'friendliai', label: 'FriendliAI', value: 'friendliai', description: 'Optimized inference' },
  { key: 'groq', label: 'Groq', value: 'groq', description: 'Ultra-fast LPU inference' },
  { key: 'nlp_cloud', label: 'NLP Cloud', value: 'nlp_cloud', description: 'Production NLP API' },
  { key: 'ollama', label: 'Ollama', value: 'ollama', description: 'Run models locally' },
  { key: 'openrouter', label: 'OpenRouter', value: 'openrouter', description: 'Unified API for many providers' },
  { key: 'perplexity', label: 'Perplexity', value: 'perplexity', description: 'Search-augmented AI' },
  { key: 'replicate', label: 'Replicate', value: 'replicate', description: 'Run ML models in the cloud' },
  { key: 'together_ai', label: 'Together AI', value: 'together_ai', description: 'Open source model hosting' },
  { key: 'voyage', label: 'Voyage', value: 'voyage', description: 'Embedding models' },
  { key: 'xai', label: 'xAI', value: 'xai', description: 'Grok models' },
];

function BasicExample() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ marginBottom: 16 }}>
        <Typography.Text>Selected: {selected ?? 'None'}</Typography.Text>
      </div>
      <Button componentId="open-modal" onClick={() => setOpen(true)}>
        Select a LiteLLM Provider
      </Button>
      <SelectorModal
        componentId="mlflow.storybook.selector-modal"
        open={open}
        onClose={() => setOpen(false)}
        onSelect={(value) => setSelected(value)}
        title="Select a LiteLLM Provider"
        items={litellmProviders}
        searchPlaceholder="Search providers..."
      />
    </div>
  );
}

export const Basic: Story = {
  render: () => <BasicExample />,
};

const countriesWithFlags: SelectorItem[] = [
  { key: 'us', label: 'United States', value: 'us', icon: <span>🇺🇸</span> },
  { key: 'uk', label: 'United Kingdom', value: 'uk', icon: <span>🇬🇧</span> },
  { key: 'de', label: 'Germany', value: 'de', icon: <span>🇩🇪</span> },
  { key: 'fr', label: 'France', value: 'fr', icon: <span>🇫🇷</span> },
  { key: 'jp', label: 'Japan', value: 'jp', icon: <span>🇯🇵</span> },
  { key: 'ca', label: 'Canada', value: 'ca', icon: <span>🇨🇦</span> },
  { key: 'au', label: 'Australia', value: 'au', icon: <span>🇦🇺</span> },
  { key: 'br', label: 'Brazil', value: 'br', icon: <span>🇧🇷</span> },
];

function WithIconsExample() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const selectedItem = countriesWithFlags.find((c) => c.value === selected);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ marginBottom: 16 }}>
        <Typography.Text>
          Selected: {selectedItem ? `${selectedItem.icon} ${selectedItem.label}` : 'None'}
        </Typography.Text>
      </div>
      <Button componentId="open-modal" onClick={() => setOpen(true)}>
        Select a Country
      </Button>
      <SelectorModal
        componentId="mlflow.storybook.country-selector"
        open={open}
        onClose={() => setOpen(false)}
        onSelect={(value) => setSelected(value)}
        title="Select a Country"
        items={countriesWithFlags}
        searchPlaceholder="Search countries..."
      />
    </div>
  );
}

export const WithIcons: Story = {
  render: () => <WithIconsExample />,
};

interface ModelItem {
  id: string;
  name: string;
  params: string;
}

const models: SelectorItem<ModelItem>[] = [
  {
    key: 'llama-70b',
    label: 'Llama 3 70B',
    value: { id: 'llama-70b', name: 'Llama 3 70B', params: '70B' },
    description: '70 billion parameters',
  },
  {
    key: 'llama-8b',
    label: 'Llama 3 8B',
    value: { id: 'llama-8b', name: 'Llama 3 8B', params: '8B' },
    description: '8 billion parameters',
  },
  {
    key: 'mistral-7b',
    label: 'Mistral 7B',
    value: { id: 'mistral-7b', name: 'Mistral 7B', params: '7B' },
    description: '7 billion parameters',
  },
  {
    key: 'mixtral-8x7b',
    label: 'Mixtral 8x7B',
    value: { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', params: '46.7B' },
    description: 'Mixture of experts - 46.7 billion parameters',
  },
];

function CustomValueExample() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<ModelItem | null>(null);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ marginBottom: 16 }}>
        <Typography.Text>Selected: {selected ? `${selected.name} (${selected.params})` : 'None'}</Typography.Text>
      </div>
      <Button componentId="open-modal" onClick={() => setOpen(true)}>
        Select a Model
      </Button>
      <SelectorModal<ModelItem>
        componentId="mlflow.storybook.model-selector"
        open={open}
        onClose={() => setOpen(false)}
        onSelect={(value) => setSelected(value)}
        title="Select a Model"
        items={models}
        searchPlaceholder="Search models..."
      />
    </div>
  );
}

export const CustomValue: Story = {
  render: () => <CustomValueExample />,
};

function CustomRenderExample() {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ marginBottom: 16 }}>
        <Typography.Text>Selected: {selected ?? 'None'}</Typography.Text>
      </div>
      <Button componentId="open-modal" onClick={() => setOpen(true)}>
        Select Provider (Custom Render)
      </Button>
      <SelectorModal
        componentId="mlflow.storybook.custom-render"
        open={open}
        onClose={() => setOpen(false)}
        onSelect={(value) => setSelected(value)}
        title="Select a Provider"
        items={litellmProviders.slice(0, 5)}
        searchPlaceholder="Search..."
        renderItem={(item, defaultContent) => (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            {defaultContent}
            <span style={{ fontSize: 12, color: '#888' }}>via LiteLLM</span>
          </div>
        )}
      />
    </div>
  );
}

export const CustomRender: Story = {
  render: () => <CustomRenderExample />,
};

function EmptyStateExample() {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ padding: 20 }}>
      <Button componentId="open-modal" onClick={() => setOpen(true)}>
        Open Empty Modal
      </Button>
      <SelectorModal
        componentId="mlflow.storybook.empty-modal"
        open={open}
        onClose={() => setOpen(false)}
        onSelect={() => {}}
        title="Select an Item"
        items={[]}
        emptyMessage="No providers available. Please check your configuration."
      />
    </div>
  );
}

export const EmptyState: Story = {
  render: () => <EmptyStateExample />,
};
