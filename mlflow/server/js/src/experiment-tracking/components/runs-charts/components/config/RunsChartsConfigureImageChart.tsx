import { useCallback } from 'react';
import type { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import { Input } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';
import { DialogCombobox } from '@databricks/design-system';
import { DialogComboboxContent } from '@databricks/design-system';
import { DialogComboboxTrigger } from '@databricks/design-system';
import { DialogComboboxOptionListCheckboxItem } from '@databricks/design-system';
import { DialogComboboxOptionList } from '@databricks/design-system';
import { useImageSliderStepMarks } from '../../hooks/useImageSliderStepMarks';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';

export const RunsChartsConfigureImageChart = ({
  previewData,
  state,
  onStateChange,
  imageKeyList,
}: {
  previewData: RunsChartsRunData[];
  imageKeyList: string[];
  state: Partial<RunsChartsImageCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const IMAGE_CONFIG_WIDTH = 275;

  const { stepMarks, maxMark, minMark } = useImageSliderStepMarks({
    data: previewData,
    selectedImageKeys: state.imageKeys || [],
  });

  const updateImageKeys = useCallback(
    (imageKeys: string[]) => {
      onStateChange((current) => {
        return { ...(current as RunsChartsImageCardConfig), imageKeys };
      });
    },
    [onStateChange],
  );

  const updateStep = useCallback(
    (step: number) => {
      onStateChange((current) => {
        return { ...(current as RunsChartsImageCardConfig), step };
      });
    },
    [onStateChange],
  );

  const { formatMessage } = useIntl();

  const handleUpdate = (imageKey: string) => {
    onStateChange((current) => {
      const currentConfig = current as RunsChartsImageCardConfig;
      if (currentConfig.imageKeys?.includes(imageKey)) {
        return {
          ...currentConfig,
          imageKeys: currentConfig.imageKeys?.filter((key) => key !== imageKey),
        };
      } else {
        return { ...currentConfig, imageKeys: [...(currentConfig.imageKeys || []), imageKey] };
      }
    });
  };

  const handleClear = () => {
    onStateChange((current) => {
      return { ...(current as RunsChartsImageCardConfig), imageKeys: [] };
    });
  };

  return (
    <>
      <RunsChartsConfigureField
        title={formatMessage({
          defaultMessage: 'Images',
          description: 'Runs charts > components > config > RunsChartsConfigureImageGrid > Images section',
        })}
      >
        <DialogCombobox
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfigureimagechart.tsx_84"
          value={state.imageKeys}
          label="Images"
          multiSelect
        >
          <DialogComboboxTrigger onClear={handleClear} minWidth={IMAGE_CONFIG_WIDTH} />
          <DialogComboboxContent matchTriggerWidth>
            <DialogComboboxOptionList>
              {imageKeyList.map((imageKey) => {
                return (
                  <DialogComboboxOptionListCheckboxItem
                    key={imageKey}
                    value={imageKey}
                    onChange={handleUpdate}
                    checked={state.imageKeys?.includes(imageKey)}
                  />
                );
              })}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </RunsChartsConfigureField>
      <RunsChartsConfigureField title="Step">
        <LineSmoothSlider
          max={maxMark}
          min={minMark}
          marks={stepMarks}
          value={state.step}
          disabled={Object.keys(stepMarks).length <= 1}
          onChange={updateStep}
        />
      </RunsChartsConfigureField>
    </>
  );
};
