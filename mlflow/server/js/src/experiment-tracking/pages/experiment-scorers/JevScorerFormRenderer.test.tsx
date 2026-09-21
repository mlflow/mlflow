import { describe, expect, it, jest } from '@jest/globals';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormProvider, useForm } from 'react-hook-form';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { Button } from '@databricks/design-system';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import JevScorerFormRenderer, { type JevScorerFormData } from './JevScorerFormRenderer';
import { SCORER_FORM_MODE, type ScorerFormMode } from './constants';
import { EndpointSelector } from '../../components/EndpointSelector';

jest.mock('../../components/EndpointSelector', () => ({ EndpointSelector: jest.fn(() => <div />) }));
jest.mock('../../components/experiment-page/hooks/useExperimentIds', () => ({ useExperimentIds: () => ['exp-123'] }));
jest.mock('@databricks/web-shared/snippet', () => ({ CodeSnippet: () => <div /> }));

const defaults: JevScorerFormData = {
  scorerType: 'jev',
  name: 'quality',
  model: 'gateway:/jev',
  question: 'Is it correct?',
  answerType: 'noul',
  criteria: '',
  threshold: '0.7',
  sampleRate: 25,
};

function TestForm({
  mode = SCORER_FORM_MODE.CREATE,
  onSubmit = jest.fn(),
}: {
  mode?: ScorerFormMode;
  onSubmit?: (data: JevScorerFormData) => void;
}) {
  const form = useForm<JevScorerFormData>({ defaultValues: defaults, mode: 'onChange' });
  return (
    <QueryClientProvider client={new QueryClient()}>
      <FormProvider {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)}>
          <JevScorerFormRenderer mode={mode} control={form.control} setValue={form.setValue} />
          <Button componentId="test.jev.submit" htmlType="submit">
            Save test
          </Button>
        </form>
      </FormProvider>
    </QueryClientProvider>
  );
}

describe('JevScorerFormRenderer', () => {
  it('restricts model selection to TypeSafe and preserves saved sample rate', async () => {
    const onSubmit = jest.fn();
    renderWithDesignSystem(<TestForm mode={SCORER_FORM_MODE.EDIT} onSubmit={onSubmit} />);
    expect(jest.mocked(EndpointSelector).mock.calls.at(-1)?.[0]).toMatchObject({ provider: 'typesafe' });
    expect(screen.queryByText('enter a model identifier')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^Name/)).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/^Question/), { target: { value: 'Updated question' } });
    await userEvent.click(screen.getByText('Save test'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0]).toMatchObject({ question: 'Updated question', threshold: '0.7', sampleRate: 25 });
  });

  it('clears incompatible criteria and threshold when switching to score', async () => {
    const onSubmit = jest.fn();
    renderWithDesignSystem(<TestForm onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Answer type' }));
    await userEvent.click(screen.getByText('Score'));
    expect(screen.queryByLabelText('Threshold')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^Criteria/)).toHaveValue('');
    fireEvent.change(screen.getByLabelText(/^Criteria/), { target: { value: '["Incorrect", "Correct"]' } });
    await userEvent.click(screen.getByText('Save test'));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      answerType: 'score',
      threshold: '',
      criteria: '["Incorrect", "Correct"]',
    });
  });

  it('prevents submission for invalid threshold and criteria', async () => {
    const onSubmit = jest.fn();
    renderWithDesignSystem(<TestForm onSubmit={onSubmit} />);
    fireEvent.change(screen.getByLabelText('Threshold'), { target: { value: '1.1' } });
    fireEvent.change(screen.getByLabelText(/^Criteria/), { target: { value: '{"yes":"invalid"}' } });
    await userEvent.click(screen.getByText('Save test'));
    expect(await screen.findByText('Enter a number between 0 and 1.')).toBeInTheDocument();
    expect(screen.getByText('Noul criteria can only describe true and false.')).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('renders the persisted configuration read-only', () => {
    renderWithDesignSystem(<TestForm mode={SCORER_FORM_MODE.DISPLAY} />);
    expect(screen.getByLabelText(/^Question/)).toHaveAttribute('readonly');
    expect(screen.getByLabelText(/^Criteria/)).toHaveAttribute('readonly');
    expect(screen.getByLabelText('Threshold')).toHaveValue(0.7);
    expect(screen.getByLabelText('Threshold')).toHaveAttribute('readonly');
    expect(screen.getByRole('switch')).toBeDisabled();
  });
});
