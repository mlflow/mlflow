import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from './ExperimentViewUtil';
import {Experiment} from "../sdk/MlflowMessages";

class ExperimentRunsTableOld extends Component {

    /*
     {
        key: runInfo.run_uuid,
        sortValue: sortValue,
        contents: ExperimentView.runInfoToRow({
          runInfo,
          onCheckbox: this.onCheckbox,
          paramKeyList,
          metricKeyList,
          paramsMap,
          metricsMap,
          tags: this.props.tagsList[idx],
          metricRanges,
          selected: !!this.state.runsSelected[runInfo.run_uuid]})
      }
     */
    static propTypes = {
        rows: PropTypes.arrayOf(PropTypes.object),
        paramKeyList: PropTypes.arrayOf(PropTypes.object),
        metricKeyList: PropTypes.arrayOf(PropTypes.object),
        onCheckAll: PropTypes.func.isRequired,
        isAllChecked: PropTypes.func.isRequired,
        onSortBy: PropTypes.func.isRequired,
        sortState: PropTypes.object.isRequired,
    };

    render() {

        const { paramKeyList, metricKeyList, rows, onCheckAll, isAllChecked, onSortBy, sortState } = this.props;
        const columns = ExperimentRunsTableOld.getColumnHeaders(paramKeyList, metricKeyList,
            onCheckAll, isAllChecked(), onSortBy, sortState);

        return (<Table hover>
            <colgroup span="7"/>
            <colgroup span={paramKeyList.length}/>
            <colgroup span={metricKeyList.length}/>
            <tbody>
            <tr>
                <th className="top-row" scope="colgroup" colSpan="5"></th>
                <th className="top-row left-border" scope="colgroup"
                    colSpan={paramKeyList.length}>Parameters
                </th>
                <th className="top-row left-border" scope="colgroup"
                    colSpan={metricKeyList.length}>Metrics
                </th>
            </tr>
            <tr>
                {columns}
            </tr>
            {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
            </tbody>
        </Table>);
    }


    static getColumnHeaders(paramKeyList, metricKeyList,
                            onCheckAll,
                            isAllCheckedBool,
                            onSortBy,
                            sortState) {

        const numParams = paramKeyList.length;
        const numMetrics = metricKeyList.length;
        const columns = [
            <th key="meta-check" className="bottom-row">
                <input type="checkbox" onChange={onCheckAll} checked={isAllCheckedBool} />
            </th>,
            ExperimentViewUtil.getHeaderCell("start_time", <span>{"Date"}</span>, true, onSortBy, sortState),
            ExperimentViewUtil.getHeaderCell("user_id", <span>{"User"}</span>, true, onSortBy, sortState),
            ExperimentViewUtil.getHeaderCell("source", <span>{"Source"}</span>, true, onSortBy, sortState),
            ExperimentViewUtil.getHeaderCell("source_version", <span>{"Version"}</span>, true, onSortBy, sortState),
        ];
        paramKeyList.forEach((paramKey, i) => {
            const className = "bottom-row "
                + (i === 0 ? "left-border " : "")
                + ExperimentViewUtil.sortedClassName(sortState, false, true, paramKey);
            columns.push(<th key={'param-' + paramKey} className={className}
                             onClick={() => onSortBy(false, true, paramKey)}>{paramKey}</th>);
        });
        if (numParams === 0) {
            columns.push(<th key="meta-param-empty" className="bottom-row left-border">(n/a)</th>);
        }

        let firstMetric = true;
        metricKeyList.forEach((metricKey) => {
            const className = "bottom-row "
                + (firstMetric ? "left-border " : "")
                + ExperimentViewUtil.sortedClassName(sortState, true, false, metricKey);
            firstMetric = false;
            columns.push(<th key={'metric-' + metricKey} className={className}
                             onClick={() => onSortBy(true, false, metricKey)}>{metricKey}</th>);
        });
        if (numMetrics === 0) {
            columns.push(<th key="meta-metric-empty" className="bottom-row left-border">(n/a)</th>);
        }

        return columns;
    }
}

// const mapStateToProps = () => {
//
// };

// export default connect(mapStateToProps)(ExperimentRunsTableOld);
export default ExperimentRunsTableOld;

