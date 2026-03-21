import { Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useMemo, useState } from 'react';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { uniq, compact } from 'lodash';
import { canEvaluateOnRun, extractRequiredInputParamsForRun } from '../../prompt-engineering/PromptEngineering.utils';
import { FormattedMessage } from 'react-intl';

const MAX_RUN_NAMES = 5;

export const useEvaluationAddNewInputsModal = () => {
  const [modalVisible, setModalVisible] = useState(false);
  const [requiredInputKeys, setRequiredInputKeys] = useState<
    {
      inputName: string;
      runNames: string[];
    }[]
  >([]);
  const [inputValues, setInputValues] = useState<Record<string, string>>({});

  const allValuesProvided = useMemo(
    () => requiredInputKeys.every(({ inputName }) => inputValues[inputName]),
    [inputValues, requiredInputKeys],
  );

  const [successCallback, setSuccessCallback] = useState<(providedParamValues: Record<string, string>) => void>(
    async () => {},
  );

  const setInputValue = useCallback((key: string, value: string) => {
    setInputValues((values) => ({ ...values, [key]: value }));
  }, []);

  const showAddNewInputsModal = useCallback(
    (runs: RunRowType[], onSuccess: (providedParamValues: Record<string, string>) => void) => {
      const requiredInputsForRuns = runs.filter(canEvaluateOnRun).map((run) => ({
        runName: run.runName,
        params: extractRequiredInputParamsForRun(run),
      }));
      const inputValuesWithRunNames = uniq(requiredInputsForRuns.map(({ params }) => params).flat()).map(
        (inputName) => ({
          inputName,
          runNames: compact(
            requiredInputsForRuns.filter((r) => r.params.includes(inputName)).map(({ runName }) => runName),
          ),
        }),
      );
      setModalVisible(true);
      setRequiredInputKeys(inputValuesWithRunNames);
      setInputValues({});
      setSuccessCallback(() => onSuccess);
    },
    [],
  );
  const { theme } = useDesignSystemTheme();

  const AddNewInputsModal = (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationaddnewinputsmodal.tsx_57"
      title={
        <FormattedMessage
          defaultMessage="Add row"
          description='Experiment page > artifact compare view > "add new row" modal title'
        />
      }
      okButtonProps={{ disabled: !allValuesProvided }}
      okText={
        <FormattedMessage
          // TODO(ML-32664): Implement "Submit and evaluate" that evaluates entire row
          defaultMessage="Submit"
          description='Experiment page > artifact compare view > "add new row" modal submit button label'
        />
      }
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description='Experiment page > artifact compare view > "add new row" modal cancel button label'
        />
      }
      onOk={() => {
        successCallback(inputValues);
        setModalVisible(false);
      }}
      visible={modalVisible}
      onCancel={() => setModalVisible(false)}
    >
      {requiredInputKeys.map(({ inputName, runNames }) => (
        <div key={inputName} css={{ marginBottom: theme.spacing.md }}>
          <Typography.Text bold>{inputName}</Typography.Text>
          <Typography.Hint css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            <FormattedMessage
              defaultMessage="Used by {runNames} {hasMore, select, true {and other runs} other {}}"
              description="Experiment page > artifact compare view > label indicating which runs are using particular input field"
              values={{
                runNames: runNames.slice(0, MAX_RUN_NAMES).join(', '),
                hasMore: runNames.length > MAX_RUN_NAMES,
              }}
            />
          </Typography.Hint>
          <div css={{ marginTop: theme.spacing.sm }}>
            <Input.TextArea
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_hooks_useevaluationaddnewinputsmodal.tsx_99"
              value={inputValues[inputName]}
              onChange={(e) => setInputValue(inputName, e.target.value)}
            />
          </div>
        </div>
      ))}
    </Modal>
  );
  return { showAddNewInputsModal, AddNewInputsModal };
};
