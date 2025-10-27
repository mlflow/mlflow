/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { connect } from 'react-redux';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';
import {
  getAllParamKeysByRunUuids,
  getAllMetricKeysByRunUuids,
  getSharedMetricKeysByRunUuids,
  getRunInfo,
} from '../reducers/Reducers';
import { isEmpty } from 'lodash';
import { CompareRunPlotContainer } from './CompareRunPlotContainer';
import { FormattedMessage } from 'react-intl';
import { Typography } from '@databricks/design-system';

type ParallelCoordinatesPlotPanelProps = {
  runUuids: string[];
  allParamKeys: string[];
  allMetricKeys: string[];
  sharedMetricKeys: string[];
  diffParamKeys: string[];
};

type ParallelCoordinatesPlotPanelState = any;

export class ParallelCoordinatesPlotPanel extends React.Component<
  ParallelCoordinatesPlotPanelProps,
  ParallelCoordinatesPlotPanelState
> {
  state = {
    // Default to select differing parameters. Sort alphabetically (to match
    // highlighted params in param table), then cap at first 3
    selectedParamKeys: this.props.diffParamKeys.sort().slice(0, 3),
    // Default to select the first metric key.
    // Note that there will be no color scaling if no metric is selected.
    selectedMetricKeys: this.props.sharedMetricKeys.slice(0, 1),
  };

  handleParamsSelectChange = (paramValues: any) => {
    this.setState({ selectedParamKeys: paramValues });
  };

  handleMetricsSelectChange = (metricValues: any) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  onClearAllSelect = () => {
    this.setState({ selectedParamKeys: [], selectedMetricKeys: [] });
  };

  render() {
    const { runUuids, allParamKeys, allMetricKeys } = this.props;
    const { selectedParamKeys, selectedMetricKeys } = this.state;
    return (
      <CompareRunPlotContainer
        controls={
          <ParallelCoordinatesPlotControls
            paramKeys={allParamKeys}
            metricKeys={allMetricKeys}
            selectedParamKeys={selectedParamKeys}
            selectedMetricKeys={selectedMetricKeys}
            handleMetricsSelectChange={this.handleMetricsSelectChange}
            handleParamsSelectChange={this.handleParamsSelectChange}
            onClearAllSelect={this.onClearAllSelect}
          />
        }
      >
        {!isEmpty(selectedParamKeys) || !isEmpty(selectedMetricKeys) ? (
          <ParallelCoordinatesPlotView
            runUuids={runUuids}
            paramKeys={selectedParamKeys}
            metricKeys={selectedMetricKeys}
          />
        ) : (
          // @ts-expect-error TS(2322): Type '(theme: any) => { padding: any; textAlign: s... Remove this comment to see the full error message
          <div css={styles.noValuesSelected} data-testid="no-values-selected">
            <Typography.Title level={2}>
              <FormattedMessage
                defaultMessage="Nothing to compare!"
                // eslint-disable-next-line max-len
                description="Header displayed in the metrics and params compare plot when no values are selected"
              />
            </Typography.Title>
            <FormattedMessage
              defaultMessage="Please select parameters and/or metrics to display the comparison."
              // eslint-disable-next-line max-len
              description="Explanation displayed in the metrics and params compare plot when no values are selected"
            />
          </div>
        )}
      </CompareRunPlotContainer>
    );
  }
}

export const getDiffParams = (allParamKeys: any, runUuids: any, paramsByRunUuid: any) => {
  const diffParamKeys: any = [];
  allParamKeys.forEach((param: any) => {
    // collect all values for this param
    const paramVals = runUuids.map(
      (runUuid: any) => paramsByRunUuid[runUuid][param] && paramsByRunUuid[runUuid][param].value,
    );
    if (!paramVals.every((x: any, i: any, arr: any) => x === arr[0])) diffParamKeys.push(param);
  });
  return diffParamKeys;
};

const mapStateToProps = (state: any, ownProps: any) => {
  const { runUuids: allRunUuids } = ownProps;

  // Filter out runUuids that do not have corresponding runInfos
  const runUuids = (allRunUuids ?? []).filter((uuid: string) => getRunInfo(uuid, state));
  const allParamKeys = getAllParamKeysByRunUuids(runUuids, state);
  const allMetricKeys = getAllMetricKeysByRunUuids(runUuids, state);
  const sharedMetricKeys = getSharedMetricKeysByRunUuids(runUuids, state);
  const { paramsByRunUuid } = state.entities;
  const diffParamKeys = getDiffParams(allParamKeys, runUuids, paramsByRunUuid);

  return {
    allParamKeys,
    allMetricKeys,
    sharedMetricKeys,
    diffParamKeys,
  };
};

const styles = {
  noValuesSelected: (theme: any) => ({
    padding: theme.spacing.md,
    textAlign: 'center',
  }),
};

// @ts-expect-error TS(2345): Argument of type 'typeof ParallelCoordinatesPlotPa... Remove this comment to see the full error message
export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
