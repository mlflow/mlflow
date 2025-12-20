import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { NavigableCombobox } from './NavigableCombobox';
import { createItem, createGroup, createTwoTierConfig, createView, createConfig } from './utils';
import type { NavigableComboboxConfig, ComboboxSelectableItem, ComboboxGroupItem } from './types';

const meta: Meta<typeof NavigableCombobox> = {
  title: 'Common/NavigableCombobox',
  component: NavigableCombobox,
  parameters: {
    layout: 'centered',
  },
  decorators: [
    (Story) => (
      <div style={{ width: 400, padding: 20 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof NavigableCombobox>;

const SimpleTemplate = (args: { config: NavigableComboboxConfig<string>; placeholder?: string }) => {
  const [value, setValue] = useState<string | null>(null);
  return (
    <div>
      <NavigableCombobox
        componentId="story-navigable-combobox"
        config={args.config}
        value={value}
        onChange={setValue}
        placeholder={args.placeholder}
      />
      <div style={{ marginTop: 16, fontSize: 14, color: '#666' }}>Selected: {value ?? '(none)'}</div>
    </div>
  );
};

export const SimpleList: Story = {
  render: () => {
    const config = createConfig<string>(
      [
        createView('main', [
          createItem('apple', 'Apple', 'apple'),
          createItem('banana', 'Banana', 'banana'),
          createItem('cherry', 'Cherry', 'cherry'),
          createItem('date', 'Date', 'date'),
          createItem('elderberry', 'Elderberry', 'elderberry'),
        ]),
      ],
      'main',
    );
    return <SimpleTemplate config={config} placeholder="Select a fruit..." />;
  },
};

export const WithGroups: Story = {
  render: () => {
    const config = createConfig<string>(
      [
        createView('main', [
          createGroup('citrus', 'Citrus Fruits', [
            createItem('orange', 'Orange', 'orange'),
            createItem('lemon', 'Lemon', 'lemon'),
            createItem('lime', 'Lime', 'lime'),
            createItem('grapefruit', 'Grapefruit', 'grapefruit'),
          ]),
          createGroup('berries', 'Berries', [
            createItem('strawberry', 'Strawberry', 'strawberry'),
            createItem('blueberry', 'Blueberry', 'blueberry'),
            createItem('raspberry', 'Raspberry', 'raspberry'),
          ]),
          createItem('apple', 'Apple', 'apple'),
          createItem('banana', 'Banana', 'banana'),
        ]),
      ],
      'main',
    );
    return <SimpleTemplate config={config} placeholder="Select a fruit..." />;
  },
};

export const TwoTierNavigation: Story = {
  render: () => {
    const config = createTwoTierConfig<string>({
      mainViewId: 'popular',
      mainItems: [
        createItem('openai', 'OpenAI', 'openai'),
        createItem('anthropic', 'Anthropic', 'anthropic'),
        createItem('google', 'Google AI', 'google'),
        createItem('mistral', 'Mistral AI', 'mistral'),
      ],
      moreViewId: 'all',
      moreViewLabel: 'All Providers (20+ more)',
      moreItems: [
        createItem('ai21', 'AI21 Labs', 'ai21'),
        createItem('cohere', 'Cohere', 'cohere'),
        createItem('huggingface', 'Hugging Face', 'huggingface'),
        createItem('replicate', 'Replicate', 'replicate'),
        createItem('together', 'Together AI', 'together'),
        createItem('anyscale', 'Anyscale', 'anyscale'),
        createItem('deepinfra', 'DeepInfra', 'deepinfra'),
        createItem('fireworks', 'Fireworks AI', 'fireworks'),
      ],
      backLabel: 'Back to popular providers',
    });
    return <SimpleTemplate config={config} placeholder="Search for a provider..." />;
  },
};

export const ProviderStyleExample: Story = {
  render: () => {
    const commonProviders: ComboboxSelectableItem<string>[] = [
      createItem('openai', 'OpenAI', 'openai'),
      createItem('anthropic', 'Anthropic', 'anthropic'),
    ];

    const azureVariants: ComboboxSelectableItem<string>[] = [
      createItem('azure', 'Azure OpenAI', 'azure'),
      createItem('azure-gov', 'Azure OpenAI (Government)', 'azure-gov'),
    ];

    const vertexVariants: ComboboxSelectableItem<string>[] = [
      createItem('vertex-ai', 'Vertex AI', 'vertex-ai'),
      createItem('vertex-ai-beta', 'Vertex AI (Beta)', 'vertex-ai-beta'),
    ];

    const litellmProviders: ComboboxSelectableItem<string>[] = [
      createItem('ai21', 'AI21 Labs', 'ai21'),
      createItem('cohere', 'Cohere', 'cohere'),
      createItem('huggingface', 'Hugging Face', 'huggingface'),
      createItem('replicate', 'Replicate', 'replicate'),
    ];

    const config = createConfig<string>(
      [
        createView('common', [
          ...commonProviders,
          createGroup('azure-group', 'Azure OpenAI', azureVariants),
          createGroup('vertex-group', 'Vertex AI', vertexVariants),
          {
            type: 'navigation',
            key: 'nav-litellm',
            label: `LiteLLM (${litellmProviders.length} providers)`,
            targetViewId: 'litellm',
            direction: 'forward',
          },
        ]),
        createView('litellm', litellmProviders, { label: 'Back to common providers', targetViewId: 'common' }),
      ],
      'common',
    );
    return <SimpleTemplate config={config} placeholder="Search for a provider..." />;
  },
};

interface ProviderMeta {
  icon: string;
  description: string;
}

const CustomRenderingExample = () => {
  const [value, setValue] = useState<string | null>(null);

  const config = createConfig<string>(
    [
      createView('main', [
        createItem('openai', 'OpenAI', 'openai', { icon: '🤖', metadata: { description: 'GPT-4, GPT-3.5' } }),
        createItem('anthropic', 'Anthropic', 'anthropic', {
          icon: '🧠',
          metadata: { description: 'Claude 3, Claude 2' },
        }),
        createItem('google', 'Google AI', 'google', { icon: '🔍', metadata: { description: 'Gemini, PaLM' } }),
        createItem('mistral', 'Mistral AI', 'mistral', { icon: '💨', metadata: { description: 'Mistral, Mixtral' } }),
      ]),
    ],
    'main',
  );

  return (
    <div>
      <NavigableCombobox
        componentId="story-custom-render"
        config={config}
        value={value}
        onChange={setValue}
        placeholder="Select a provider..."
        renderItem={(item, defaultRender) => {
          if (item.type !== 'item') return defaultRender;
          const meta = item.metadata as ProviderMeta | undefined;
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 20 }}>{meta?.icon}</span>
              <div>
                <div style={{ fontWeight: 500 }}>{item.label}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{meta?.description}</div>
              </div>
            </div>
          );
        }}
      />
      <div style={{ marginTop: 16, fontSize: 14, color: '#666' }}>Selected: {value ?? '(none)'}</div>
    </div>
  );
};

export const WithCustomRendering: Story = {
  render: () => <CustomRenderingExample />,
};

const ValidationErrorExample = () => {
  const [value, setValue] = useState<string | null>(null);
  const config = createConfig<string>(
    [createView('main', [createItem('apple', 'Apple', 'apple'), createItem('banana', 'Banana', 'banana')])],
    'main',
  );

  return (
    <NavigableCombobox
      componentId="story-error"
      config={config}
      value={value}
      onChange={setValue}
      placeholder="Select a fruit..."
      error="This field is required"
    />
  );
};

export const WithValidationError: Story = {
  render: () => <ValidationErrorExample />,
};

export const Disabled: Story = {
  render: () => {
    const config = createConfig<string>(
      [createView('main', [createItem('apple', 'Apple', 'apple'), createItem('banana', 'Banana', 'banana')])],
      'main',
    );

    return (
      <NavigableCombobox
        componentId="story-disabled"
        config={config}
        value="apple"
        onChange={() => {}}
        placeholder="Select a fruit..."
        disabled
      />
    );
  },
};
