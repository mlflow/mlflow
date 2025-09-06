import { useEvaluationAddNewInputsModal } from './useEvaluationAddNewInputsModal';
import { act, renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { createParamFieldName } from '../../experiment-page/utils/experimentPage.column-utils';
import { useEffect } from 'react';
import userEvent from '@testing-library/user-event';
import { MLFLOW_RUN_SOURCE_TYPE_TAG, MLflowRunSourceType } from '../../../constants';

describe('useEvaluationAddNewInputsModal', () => {
  const renderHookResult = (runs: RunRowType[], onSuccess: (providedParamValues: Record<string, string>) => void) => {
    const Component = () => {
      const { AddNewInputsModal, showAddNewInputsModal } = useEvaluationAddNewInputsModal();
      useEffect(() => {
        showAddNewInputsModal(runs, onSuccess);
      }, [showAddNewInputsModal]);
      return <>{AddNewInputsModal}</>;
    };

    return renderWithIntl(<Component />);
  };

  it('should properly calculate input field names for visible runs', async () => {
    const runA = {
      runName: 'run A',
      params: [
        { key: 'model_route', value: 'some-route' },
        {
          key: 'prompt_template',
          value: 'This is run A with {{input_a}} and {{input_b}} template',
        },
      ],
      tags: {
        [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
          key: MLFLOW_RUN_SOURCE_TYPE_TAG,
          value: MLflowRunSourceType.PROMPT_ENGINEERING,
        },
      },
    } as any;

    const runB = {
      runName: 'run B',
      params: [
        { key: 'model_route', value: 'some-route' },
        {
          key: 'prompt_template',
          value: 'This is run B with {{input_b}} and {{input_c}} template',
        },
      ],
      tags: {
        [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
          key: MLFLOW_RUN_SOURCE_TYPE_TAG,
          value: MLflowRunSourceType.PROMPT_ENGINEERING,
        },
      },
    };

    const onSuccess = jest.fn();

    renderHookResult([runA, runB], onSuccess);

    expect(
      screen.getByText((_, element) => Boolean(element?.textContent?.trim().match(/^input_a\s?Used by run A$/))),
    ).toBeInTheDocument();

    expect(
      screen.getByText((_, element) => Boolean(element?.textContent?.trim().match(/^input_b\s?Used by run A, run B$/))),
    ).toBeInTheDocument();

    expect(
      screen.getByText((_, element) => Boolean(element?.textContent?.trim().match(/^input_c\s?Used by run B$/))),
    ).toBeInTheDocument();

    // Type in data for two inputs, leave input_b empty
    act(() => screen.getAllByRole<HTMLTextAreaElement>('textbox')[0].focus());
    await userEvent.paste('val_a');
    act(() => screen.getAllByRole<HTMLTextAreaElement>('textbox')[2].focus());
    await userEvent.paste('val_c');

    expect(screen.getByRole('button', { name: 'Submit' })).toBeDisabled();

    // Fill in missing input
    act(() => screen.getAllByRole<HTMLTextAreaElement>('textbox')[1].focus());
    await userEvent.paste('val_b');

    expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled();

    await userEvent.click(screen.getByRole('button', { name: 'Submit' }));

    // Assert returned data
    expect(onSuccess).toHaveBeenCalledWith({
      input_a: 'val_a',
      input_b: 'val_b',
      input_c: 'val_c',
    });
  });
});
