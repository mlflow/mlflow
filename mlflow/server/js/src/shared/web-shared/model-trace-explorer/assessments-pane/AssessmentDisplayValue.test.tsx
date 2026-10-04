import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';

import { DesignSystemProvider } from '@databricks/design-system';

import { AssessmentDisplayValue } from './AssessmentDisplayValue';

const renderValue = (jsonValue: string, booleanDisplayMode?: 'true-false' | 'pass-fail') =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AssessmentDisplayValue jsonValue={jsonValue} booleanDisplayMode={booleanDisplayMode} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AssessmentDisplayValue', () => {
  test.each(['true', '"true"'])('preserves %s as True by default', (jsonValue) => {
    renderValue(jsonValue);

    expect(screen.getByText('True')).toBeInTheDocument();
    expect(screen.queryByText('Pass')).not.toBeInTheDocument();
  });

  test.each(['false', '"false"'])('preserves %s as False by default', (jsonValue) => {
    renderValue(jsonValue);

    expect(screen.getByText('False')).toBeInTheDocument();
    expect(screen.queryByText('Fail')).not.toBeInTheDocument();
  });

  test.each([
    ['true', 'Pass'],
    ['"true"', 'Pass'],
    ['false', 'Fail'],
    ['"false"', 'Fail'],
  ])('renders %s as %s when pass/fail semantics are explicit', (jsonValue, expectedLabel) => {
    renderValue(jsonValue, 'pass-fail');

    expect(screen.getByText(expectedLabel)).toBeInTheDocument();
  });
});
