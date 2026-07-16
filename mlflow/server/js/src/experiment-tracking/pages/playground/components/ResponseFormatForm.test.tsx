import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ResponseFormatType } from '../types';
import { ResponseFormatForm } from './ResponseFormatForm';

// Monaco does not render in jsdom; stand the editor in with a labelled textarea.
jest.mock('../../experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor', () => ({
  LazyJsonRecordEditor: ({
    ariaLabel,
    value,
    onChange,
    errorMessage,
  }: {
    ariaLabel: string;
    value: string;
    onChange: (next: string) => void;
    errorMessage?: string;
  }) => (
    <div>
      <textarea aria-label={ariaLabel} value={value} onChange={(event) => onChange(event.target.value)} />
      {errorMessage ? <div role="alert">{errorMessage}</div> : null}
    </div>
  ),
}));

interface RenderProps {
  type?: ResponseFormatType;
  schemaText?: string;
  schemaError?: string | null;
}

const renderForm = ({ type = 'text', schemaText = '', schemaError }: RenderProps = {}) => {
  const onTypeChange = jest.fn<(next: ResponseFormatType) => void>();
  const onSchemaChange = jest.fn<(next: string) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ResponseFormatForm
          type={type}
          onTypeChange={onTypeChange}
          schemaText={schemaText}
          onSchemaChange={onSchemaChange}
          schemaError={schemaError}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onTypeChange, onSchemaChange };
};

describe('ResponseFormatForm', () => {
  it('does not render the schema textarea for text or json_object', () => {
    const { unmount } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ResponseFormatForm type="text" onTypeChange={jest.fn()} schemaText="" onSchemaChange={jest.fn()} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.queryByLabelText('Schema')).not.toBeInTheDocument();
    unmount();

    renderForm({ type: 'json_object' });
    expect(screen.queryByLabelText('Schema')).not.toBeInTheDocument();
  });

  it('renders the schema textarea when type is json_schema', () => {
    renderForm({ type: 'json_schema' });
    expect(screen.getByLabelText('Schema')).toBeInTheDocument();
  });

  it('forwards type changes via onTypeChange', async () => {
    const { onTypeChange } = renderForm({ type: 'text' });
    await userEvent.click(screen.getByRole('radio', { name: 'JSON' }));
    expect(onTypeChange).toHaveBeenLastCalledWith('json_object');
  });

  it('renders the schema parse error when present', () => {
    renderForm({ type: 'json_schema', schemaText: '{not-json', schemaError: 'Unexpected token' });
    expect(screen.getByText('Unexpected token')).toBeInTheDocument();
  });
});
