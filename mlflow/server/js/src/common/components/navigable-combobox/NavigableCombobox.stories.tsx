import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { DesignSystemProvider, Typography } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { NavigableCombobox } from './NavigableCombobox';
import type { NavigableComboboxConfig, ComboboxModalTriggerItem } from './types';
import type { SelectorItem } from '../selector-modal/types';

const meta: Meta<typeof NavigableCombobox> = {
  title: 'Common/NavigableCombobox',
  component: NavigableCombobox,
  decorators: [
    (Story) => (
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <div style={{ padding: 20, maxWidth: 400 }}>
            <Story />
          </div>
        </IntlProvider>
      </DesignSystemProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof NavigableCombobox>;

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

const providerConfig: NavigableComboboxConfig = {
  initialViewId: 'main',
  views: [
    {
      id: 'main',
      items: [
        {
          type: 'group',
          key: 'openai-group',
          label: 'OpenAI',
          backLabel: 'Back to providers',
          children: [
            { type: 'item', key: 'openai', label: 'OpenAI', value: 'openai' },
            { type: 'item', key: 'azure', label: 'Azure OpenAI', value: 'azure' },
          ],
        },
        { type: 'item', key: 'anthropic', label: 'Anthropic', value: 'anthropic' },
        {
          type: 'group',
          key: 'google-group',
          label: 'Google AI',
          backLabel: 'Back to providers',
          children: [
            { type: 'item', key: 'google', label: 'Google AI', value: 'google' },
            { type: 'item', key: 'vertex', label: 'Vertex AI', value: 'vertex' },
          ],
        },
        { type: 'item', key: 'cohere', label: 'Cohere', value: 'cohere' },
        { type: 'item', key: 'mistral', label: 'Mistral', value: 'mistral' },
        { type: 'item', key: 'huggingface', label: 'Hugging Face', value: 'huggingface' },
        { type: 'item', key: 'bedrock', label: 'AWS Bedrock', value: 'bedrock' },
        {
          type: 'modal-trigger',
          key: 'others',
          label: 'Others',
          modalTitle: 'Select a LiteLLM Provider',
          modalSearchPlaceholder: 'Search providers...',
          modalItems: litellmProviders,
        } as ComboboxModalTriggerItem,
      ],
    },
  ],
};

function ProviderSelectorExample() {
  const [value, setValue] = useState<string | null>(null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Typography.Text>Selected: {value ?? 'None'}</Typography.Text>
      <NavigableCombobox
        componentId="mlflow.storybook.provider-selector"
        config={providerConfig}
        value={value}
        onChange={setValue}
        placeholder="Select a provider..."
      />
    </div>
  );
}

export const ProviderSelectorWithModalTrigger: Story = {
  render: () => <ProviderSelectorExample />,
};

const simpleConfig: NavigableComboboxConfig = {
  initialViewId: 'main',
  views: [
    {
      id: 'main',
      items: [
        { type: 'item', key: 'option1', label: 'Option 1', value: 'option1' },
        { type: 'item', key: 'option2', label: 'Option 2', value: 'option2' },
        { type: 'item', key: 'option3', label: 'Option 3', value: 'option3' },
      ],
    },
  ],
};

function SimpleExample() {
  const [value, setValue] = useState<string | null>(null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Typography.Text>Selected: {value ?? 'None'}</Typography.Text>
      <NavigableCombobox
        componentId="mlflow.storybook.simple"
        config={simpleConfig}
        value={value}
        onChange={setValue}
        placeholder="Select an option..."
      />
    </div>
  );
}

export const Simple: Story = {
  render: () => <SimpleExample />,
};

const groupedConfig: NavigableComboboxConfig = {
  initialViewId: 'main',
  views: [
    {
      id: 'main',
      items: [
        {
          type: 'group',
          key: 'fruits',
          label: 'Fruits',
          backLabel: 'Back',
          children: [
            { type: 'item', key: 'apple', label: 'Apple', value: 'apple' },
            { type: 'item', key: 'banana', label: 'Banana', value: 'banana' },
            { type: 'item', key: 'orange', label: 'Orange', value: 'orange' },
          ],
        },
        {
          type: 'group',
          key: 'vegetables',
          label: 'Vegetables',
          backLabel: 'Back',
          children: [
            { type: 'item', key: 'carrot', label: 'Carrot', value: 'carrot' },
            { type: 'item', key: 'broccoli', label: 'Broccoli', value: 'broccoli' },
            { type: 'item', key: 'spinach', label: 'Spinach', value: 'spinach' },
          ],
        },
      ],
    },
  ],
};

function GroupedExample() {
  const [value, setValue] = useState<string | null>(null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Typography.Text>Selected: {value ?? 'None'}</Typography.Text>
      <NavigableCombobox
        componentId="mlflow.storybook.grouped"
        config={groupedConfig}
        value={value}
        onChange={setValue}
        placeholder="Select food..."
      />
    </div>
  );
}

export const WithGroups: Story = {
  render: () => <GroupedExample />,
};

function WithDescriptionExample() {
  const [value, setValue] = useState<string | null>(null);

  const configWithDescriptions: NavigableComboboxConfig = {
    initialViewId: 'main',
    views: [
      {
        id: 'main',
        items: [
          {
            type: 'item',
            key: 'openai',
            label: 'OpenAI',
            value: 'openai',
            metadata: { description: 'GPT-4, GPT-3.5 Turbo, and more' },
          },
          {
            type: 'item',
            key: 'anthropic',
            label: 'Anthropic',
            value: 'anthropic',
            metadata: { description: 'Claude 3 family of models' },
          },
          {
            type: 'item',
            key: 'google',
            label: 'Google AI',
            value: 'google',
            metadata: { description: 'Gemini and PaLM models' },
          },
        ],
      },
    ],
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Typography.Text>Selected: {value ?? 'None'}</Typography.Text>
      <NavigableCombobox
        componentId="mlflow.storybook.with-description"
        config={configWithDescriptions}
        value={value}
        onChange={setValue}
        placeholder="Select a provider..."
        renderItem={(item, defaultContent) => {
          if (item.type === 'item' && item.metadata?.['description']) {
            return (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {defaultContent}
                <Typography.Text color="secondary" style={{ fontSize: 12, marginTop: 2 }}>
                  {item.metadata['description'] as string}
                </Typography.Text>
              </div>
            );
          }
          return defaultContent;
        }}
      />
    </div>
  );
}

export const WithCustomRendering: Story = {
  render: () => <WithDescriptionExample />,
};

function SearchExample() {
  const [value, setValue] = useState<string | null>(null);

  const manyItemsConfig: NavigableComboboxConfig = {
    initialViewId: 'main',
    views: [
      {
        id: 'main',
        items: Array.from({ length: 20 }, (_, i) => ({
          type: 'item' as const,
          key: `item-${i}`,
          label: `Item ${i + 1}`,
          value: `item-${i}`,
        })),
      },
    ],
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Typography.Text>Selected: {value ?? 'None'}</Typography.Text>
      <Typography.Text color="secondary" style={{ fontSize: 12 }}>
        Type to filter items
      </Typography.Text>
      <NavigableCombobox
        componentId="mlflow.storybook.search"
        config={manyItemsConfig}
        value={value}
        onChange={setValue}
        placeholder="Search items..."
      />
    </div>
  );
}

export const SearchFiltering: Story = {
  render: () => <SearchExample />,
};

function ModalHoverStylesExample() {
  const [tertiaryValue, setTertiaryValue] = useState<string | null>(null);
  const [defaultValue, setDefaultValue] = useState<string | null>(null);
  const [tableValue, setTableValue] = useState<string | null>(null);

  const modalItems: SelectorItem[] = [
    { key: 'option1', label: 'Option 1', value: 'option1', description: 'First option' },
    { key: 'option2', label: 'Option 2', value: 'option2', description: 'Second option' },
    { key: 'option3', label: 'Option 3', value: 'option3', description: 'Third option' },
  ];

  const makeConfig = (): NavigableComboboxConfig => ({
    initialViewId: 'main',
    views: [
      {
        id: 'main',
        items: [
          { type: 'item', key: 'inline', label: 'Inline Option', value: 'inline' },
          {
            type: 'modal-trigger',
            key: 'more',
            label: 'More Options',
            modalTitle: 'Select an Option',
            modalItems,
          } as ComboboxModalTriggerItem,
        ],
      },
    ],
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div>
        <Typography.Text bold>Tertiary Hover (default - subtle gray)</Typography.Text>
        <div style={{ marginTop: 8 }}>
          <NavigableCombobox
            componentId="mlflow.storybook.hover-tertiary"
            config={makeConfig()}
            value={tertiaryValue}
            onChange={setTertiaryValue}
            placeholder="Click 'More Options' to see tertiary hover..."
            modalHoverStyle="tertiary"
          />
        </div>
      </div>
      <div>
        <Typography.Text bold>Default Hover (button default)</Typography.Text>
        <div style={{ marginTop: 8 }}>
          <NavigableCombobox
            componentId="mlflow.storybook.hover-default"
            config={makeConfig()}
            value={defaultValue}
            onChange={setDefaultValue}
            placeholder="Click 'More Options' to see default hover..."
            modalHoverStyle="default"
          />
        </div>
      </div>
      <div>
        <Typography.Text bold>Table Selection Hover (light blue)</Typography.Text>
        <div style={{ marginTop: 8 }}>
          <NavigableCombobox
            componentId="mlflow.storybook.hover-table"
            config={makeConfig()}
            value={tableValue}
            onChange={setTableValue}
            placeholder="Click 'More Options' to see table selection hover..."
            modalHoverStyle="tableSelection"
          />
        </div>
      </div>
    </div>
  );
}

export const ModalHoverStyles: Story = {
  render: () => <ModalHoverStylesExample />,
};
