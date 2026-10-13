import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { LabelSchemaPreview } from './LabelSchemaPreview';
import { DEFAULT_FORM_VALUES, type LabelSchemaFormData } from './labelSchemaFormUtils';

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('LabelSchemaPreview', () => {
  it('renders name, instruction, and pass-fail widget for a pass_fail schema', () => {
    const formData: LabelSchemaFormData = {
      ...DEFAULT_FORM_VALUES,
      name: 'Is the answer correct?',
      instruction: 'Mark Correct if accurate.',
      inputKind: 'pass_fail',
      passFailPositiveLabel: 'Correct',
      passFailNegativeLabel: 'Incorrect',
    };
    renderWithProviders(<LabelSchemaPreview formData={formData} />);
    expect(screen.getByText('Is the answer correct?')).toBeInTheDocument();
    expect(screen.getByText('Mark Correct if accurate.')).toBeInTheDocument();
    expect(screen.getByText('Correct')).toBeInTheDocument();
    expect(screen.getByText('Incorrect')).toBeInTheDocument();
  });

  it('renders the categorical widget (DialogCombobox) for a categorical schema', () => {
    const formData: LabelSchemaFormData = {
      ...DEFAULT_FORM_VALUES,
      name: 'Severity',
      inputKind: 'categorical',
      categoricalOptions: ['low', 'medium', 'high'],
    };
    renderWithProviders(<LabelSchemaPreview formData={formData} />);
    expect(screen.getByText('Severity')).toBeInTheDocument();
    // The categorical widget renders a DialogCombobox trigger. Assert
    // the trigger button is in the DOM so a regression that strips
    // the input widget would surface here rather than passing on the
    // name text alone.
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('renders the numeric widget (number input) for a numeric schema', () => {
    const formData: LabelSchemaFormData = {
      ...DEFAULT_FORM_VALUES,
      name: 'Rating',
      inputKind: 'numeric',
      numericMinValue: '1',
      numericMaxValue: '5',
    };
    renderWithProviders(<LabelSchemaPreview formData={formData} />);
    expect(screen.getByText('Rating')).toBeInTheDocument();
    // The numeric widget renders an <input type="number"> which exposes
    // the `spinbutton` role. Assert it's present + the schema's bounds
    // are wired through so a regression that drops the widget or the
    // min/max attributes would surface here.
    const numberInput = screen.getByRole('spinbutton') as HTMLInputElement;
    expect(numberInput).toBeInTheDocument();
    expect(numberInput.min).toBe('1');
    expect(numberInput.max).toBe('5');
  });

  it('renders the text widget (textarea) for a text schema', () => {
    const formData: LabelSchemaFormData = {
      ...DEFAULT_FORM_VALUES,
      name: 'Expected answer',
      inputKind: 'text',
      textMaxLength: '200',
    };
    renderWithProviders(<LabelSchemaPreview formData={formData} />);
    expect(screen.getByText('Expected answer')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('shows a placeholder when name is blank', () => {
    const formData: LabelSchemaFormData = {
      ...DEFAULT_FORM_VALUES,
      name: '',
      inputKind: 'pass_fail',
      passFailPositiveLabel: 'A',
      passFailNegativeLabel: 'B',
    };
    renderWithProviders(<LabelSchemaPreview formData={formData} />);
    expect(screen.getByText('(no name yet)')).toBeInTheDocument();
  });
});
