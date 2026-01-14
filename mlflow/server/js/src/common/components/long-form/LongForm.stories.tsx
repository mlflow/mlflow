import type { Meta, StoryObj } from '@storybook/react';
import { DesignSystemProvider, Input, Button, Typography } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { LongFormSection } from './LongFormSection';
import { LongFormSummary } from './LongFormSummary';

const meta: Meta<typeof LongFormSection> = {
  title: 'Common/LongForm',
  component: LongFormSection,
  decorators: [
    (Story) => (
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <div style={{ padding: 20, maxWidth: 900 }}>
            <Story />
          </div>
        </IntlProvider>
      </DesignSystemProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof LongFormSection>;

export const BasicSection: Story = {
  render: () => (
    <LongFormSection title="Name">
      <Input componentId="storybook.long-form.name" placeholder="Enter a name..." />
    </LongFormSection>
  ),
};

export const SectionWithoutDivider: Story = {
  render: () => (
    <LongFormSection title="Description" hideDivider>
      <Input componentId="storybook.long-form.description" placeholder="Enter a description..." />
    </LongFormSection>
  ),
};

export const MultipleSections: Story = {
  render: () => (
    <div>
      <LongFormSection title="Name">
        <Input componentId="storybook.long-form.name" placeholder="Enter a name..." />
      </LongFormSection>
      <LongFormSection title="Provider">
        <Button componentId="storybook.long-form.provider">Select a provider...</Button>
      </LongFormSection>
      <LongFormSection title="Model" hideDivider>
        <Button componentId="storybook.long-form.model">Select a model...</Button>
      </LongFormSection>
    </div>
  ),
};

export const SummaryPanel: Story = {
  render: () => (
    <div style={{ maxWidth: 360 }}>
      <LongFormSummary title="Summary">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <Typography.Text bold color="secondary">
              Provider
            </Typography.Text>
            <Typography.Text>OpenAI</Typography.Text>
          </div>
          <div>
            <Typography.Text bold color="secondary">
              Model
            </Typography.Text>
            <Typography.Text>gpt-4</Typography.Text>
          </div>
        </div>
      </LongFormSummary>
    </div>
  ),
};

export const FormWithSidebar: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 24 }}>
      <div style={{ flexGrow: 1, maxWidth: 600 }}>
        <LongFormSection title="Name">
          <Input componentId="storybook.long-form.name" placeholder="my-endpoint" />
        </LongFormSection>
        <LongFormSection title="Model" hideDivider>
          <Button componentId="storybook.long-form.model">Select a model...</Button>
        </LongFormSection>
      </div>
      <div style={{ width: 300, flexShrink: 0 }}>
        <LongFormSummary title="Summary">
          <Typography.Text color="secondary">Configure your endpoint to see a summary.</Typography.Text>
        </LongFormSummary>
      </div>
    </div>
  ),
};
