/**
 * TODO: implement actual UI for this modal, it's a crude placeholder with minimal logic for now
 */
import { Input, Modal } from '@databricks/design-system';
import { Interpolation, Theme } from '@emotion/react';
import { useState } from 'react';
import { CompareRunsChartSetup } from '../../types';
import { CompareRunsScatterPlot } from './charts/CompareRunsScatterPlot';

export const ConfigureChartModal = ({
  onCancel,
  onSubmit,
}: {
  onCancel: () => void;
  onSubmit: (formData: Pick<CompareRunsChartSetup, 'type' | 'metricKey'>) => void;
}) => {
  const [currentFormState, setCurrentFormState] = useState<
    Pick<CompareRunsChartSetup, 'type' | 'metricKey'>
  >({
    type: 'BAR',
    metricKey: 'metric_1',
  });

  return (
    <Modal
      visible
      onCancel={onCancel}
      onOk={() => onSubmit(currentFormState)}
      title={'[TODO] Add new chart'}
      cancelText='Cancel'
      okText='Confirm'
      size='wide'
    >
      <div css={styles.wrapper}>
        <div>
          <div css={styles.field}>
            <span>Type:</span>
            <Input
              value={currentFormState.type}
              onChange={(e) => setCurrentFormState((state) => ({ ...state, type: e.target.value }))}
            />
          </div>
          <div css={styles.field}>
            <span>Metric key:</span>
            <Input
              value={currentFormState.metricKey}
              onChange={(e) =>
                setCurrentFormState((state) => ({ ...state, metricKey: e.target.value }))
              }
            />
          </div>
        </div>
        <div css={styles.chartWrapper}>
          <CompareRunsScatterPlot
            runsData={[]}
            xAxis={{ key: 'TODO', type: 'METRIC' }}
            yAxis={{ key: 'TODO', type: 'METRIC' }}
          />
        </div>
      </div>
    </Modal>
  );
};

const styles = {
  wrapper: {
    display: 'grid',
    gridTemplateColumns: '200px 1fr',
    gap: 32,
  } as Interpolation<Theme>,
  field: {
    display: 'grid',
    gridTemplateColumns: '80px 1fr',
    marginBottom: 16,
  } as Interpolation<Theme>,
  chartWrapper: {
    height: 400,
  },
};
