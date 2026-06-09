import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { LabelSchemaInputCategorical } from './LabelSchemaInputCategorical';
import { LabelSchemaInputNumeric } from './LabelSchemaInputNumeric';
import { LabelSchemaInputPassFail } from './LabelSchemaInputPassFail';
import { LabelSchemaInputRenderer } from './LabelSchemaInputRenderer';

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('LabelSchemaInputPassFail', () => {
  it('renders positive and negative labels', () => {
    renderWithProviders(
      <LabelSchemaInputPassFail
        input={{ positive_label: 'Correct', negative_label: 'Incorrect' }}
        value={null}
        onChange={jest.fn()}
        componentId="test.pass-fail"
      />,
    );
    expect(screen.getByText('Correct')).toBeInTheDocument();
    expect(screen.getByText('Incorrect')).toBeInTheDocument();
  });

  it('fires onChange(true) when the positive button is clicked', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputPassFail
        input={{ positive_label: 'Correct', negative_label: 'Incorrect' }}
        value={null}
        onChange={onChange}
        componentId="test.pass-fail"
      />,
    );
    fireEvent.click(screen.getByText('Correct'));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it('fires onChange(false) when the negative button is clicked', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputPassFail
        input={{ positive_label: 'Correct', negative_label: 'Incorrect' }}
        value={null}
        onChange={onChange}
        componentId="test.pass-fail"
      />,
    );
    fireEvent.click(screen.getByText('Incorrect'));
    expect(onChange).toHaveBeenCalledWith(false);
  });
});

describe('LabelSchemaInputNumeric', () => {
  it('renders min/max bounds on the input', () => {
    renderWithProviders(
      <LabelSchemaInputNumeric
        input={{ min_value: 1, max_value: 5 }}
        value={3}
        onChange={jest.fn()}
        componentId="test.numeric"
      />,
    );
    const input = screen.getByRole('spinbutton') as HTMLInputElement;
    expect(input.value).toEqual('3');
    expect(input.min).toEqual('1');
    expect(input.max).toEqual('5');
  });

  it('clears to null when the user empties the field', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputNumeric
        input={{ min_value: 1, max_value: 5 }}
        value={3}
        onChange={onChange}
        componentId="test.numeric"
      />,
    );
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('forwards parsed numeric values', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputNumeric
        input={{ min_value: 1, max_value: 5 }}
        value={null}
        onChange={onChange}
        componentId="test.numeric"
      />,
    );
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '4' } });
    expect(onChange).toHaveBeenCalledWith(4);
  });

  it('omits min/max attributes when not set on the input', () => {
    renderWithProviders(
      <LabelSchemaInputNumeric input={{}} value={null} onChange={jest.fn()} componentId="test.numeric" />,
    );
    const input = screen.getByRole('spinbutton') as HTMLInputElement;
    expect(input.min).toEqual('');
    expect(input.max).toEqual('');
  });
});

describe('LabelSchemaInputCategorical', () => {
  it('renders single-select by default', () => {
    renderWithProviders(
      <LabelSchemaInputCategorical
        input={{ options: ['low', 'medium', 'high'] }}
        value={null}
        onChange={jest.fn()}
        componentId="test.categorical"
        label="Severity"
      />,
    );
    expect(screen.getByRole('combobox', { name: /Severity/ })).toBeInTheDocument();
  });

  it('renders multi-select when input.multi_select is true', () => {
    renderWithProviders(
      <LabelSchemaInputCategorical
        input={{ options: ['low', 'high'], multi_select: true }}
        value={[]}
        onChange={jest.fn()}
        componentId="test.categorical"
        label="Severity"
      />,
    );
    expect(screen.getByRole('combobox', { name: /Severity/ })).toBeInTheDocument();
  });

  it('fires onChange(option) when a single-select option is clicked', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputCategorical
        input={{ options: ['low', 'medium', 'high'] }}
        value={null}
        onChange={onChange}
        componentId="test.categorical"
        label="Severity"
      />,
    );
    fireEvent.click(screen.getByRole('combobox', { name: /Severity/ }));
    fireEvent.click(screen.getByText('medium'));
    expect(onChange).toHaveBeenCalledWith('medium');
  });

  it('fires onChange([option]) when a multi-select option is clicked from empty', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputCategorical
        input={{ options: ['low', 'high'], multi_select: true }}
        value={[]}
        onChange={onChange}
        componentId="test.categorical"
        label="Severity"
      />,
    );
    fireEvent.click(screen.getByRole('combobox', { name: /Severity/ }));
    fireEvent.click(screen.getByText('low'));
    expect(onChange).toHaveBeenCalledWith(['low']);
  });
});

describe('LabelSchemaInputRenderer', () => {
  it('dispatches to the pass-fail widget when input.pass_fail is set', () => {
    renderWithProviders(
      <LabelSchemaInputRenderer
        input={{ pass_fail: { positive_label: 'Yes', negative_label: 'No' } }}
        value={null}
        onChange={jest.fn()}
        componentId="test.dispatcher"
      />,
    );
    expect(screen.getByText('Yes')).toBeInTheDocument();
    expect(screen.getByText('No')).toBeInTheDocument();
  });

  it('dispatches to the numeric widget when input.numeric is set', () => {
    renderWithProviders(
      <LabelSchemaInputRenderer
        input={{ numeric: { min_value: 1, max_value: 5 } }}
        value={2}
        onChange={jest.fn()}
        componentId="test.dispatcher"
      />,
    );
    expect(screen.getByRole('spinbutton')).toBeInTheDocument();
  });

  it('dispatches to the categorical widget when input.categorical is set', () => {
    renderWithProviders(
      <LabelSchemaInputRenderer
        input={{ categorical: { options: ['a', 'b'] } }}
        value={null}
        onChange={jest.fn()}
        componentId="test.dispatcher"
        label="Pick one"
      />,
    );
    expect(screen.getByRole('combobox', { name: /Pick one/ })).toBeInTheDocument();
  });

  it('dispatches to the text widget when input.text is set', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputRenderer
        input={{ text: { max_length: 200 } }}
        value={null}
        onChange={onChange}
        componentId="test.dispatcher"
      />,
    );
    const textbox = screen.getByRole('textbox');
    expect(textbox).toBeInTheDocument();
    fireEvent.change(textbox, { target: { value: 'hello' } });
    expect(onChange).toHaveBeenCalledWith('hello');
  });

  it('renders an error Alert when the input wrapper has no variant set', () => {
    renderWithProviders(
      <LabelSchemaInputRenderer input={{}} value={null} onChange={jest.fn()} componentId="test.dispatcher" />,
    );
    expect(screen.getByText(/Invalid label schema/i)).toBeInTheDocument();
  });

  it('propagates a null value upward when the numeric widget is cleared', () => {
    const onChange = jest.fn();
    renderWithProviders(
      <LabelSchemaInputRenderer
        input={{ numeric: { min_value: 1, max_value: 5 } }}
        value={3}
        onChange={onChange}
        componentId="test.dispatcher"
      />,
    );
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
