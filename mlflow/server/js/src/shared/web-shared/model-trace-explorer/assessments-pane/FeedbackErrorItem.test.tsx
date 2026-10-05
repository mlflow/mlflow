import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { FeedbackErrorItem } from './FeedbackErrorItem';
import { ModelTraceExplorerRunJudgesContextProvider } from '../contexts/RunJudgesContext';
import type { AssessmentError } from '../ModelTrace.types';

const RAW_ERROR: AssessmentError = {
  error_code: 'SCORER_ERROR',
  error_message: 'TypeSafe judge models do not support trace-based evaluation.',
};

const renderItem = (ui: React.ReactNode) =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">{ui}</IntlProvider>
    </DesignSystemProvider>,
  );

describe('FeedbackErrorItem', () => {
  it('shows the raw error message when no formatter is provided', () => {
    renderItem(<FeedbackErrorItem error={RAW_ERROR} />);
    expect(screen.getByText(RAW_ERROR.error_message as string)).toBeInTheDocument();
  });

  it('shows the feature-provided friendly message when the formatter matches', () => {
    const formatErrorMessage = (message: string) =>
      message.includes('trace-based evaluation') ? 'Rewrite the instructions without {{ trace }}.' : null;

    renderItem(
      <ModelTraceExplorerRunJudgesContextProvider formatErrorMessage={formatErrorMessage}>
        <FeedbackErrorItem error={RAW_ERROR} />
      </ModelTraceExplorerRunJudgesContextProvider>,
    );

    expect(screen.getByText('Rewrite the instructions without {{ trace }}.')).toBeInTheDocument();
    expect(screen.queryByText(RAW_ERROR.error_message as string)).not.toBeInTheDocument();
  });

  it('falls back to the raw message when the formatter does not match', () => {
    renderItem(
      <ModelTraceExplorerRunJudgesContextProvider formatErrorMessage={() => null}>
        <FeedbackErrorItem error={RAW_ERROR} />
      </ModelTraceExplorerRunJudgesContextProvider>,
    );
    expect(screen.getByText(RAW_ERROR.error_message as string)).toBeInTheDocument();
  });
});
