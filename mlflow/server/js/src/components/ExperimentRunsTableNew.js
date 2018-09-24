import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from "./ExperimentViewUtil";

class ExperimentRunsTableNew extends Component {

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
      const columns = [
        <th key="meta-check" className="bottom-row">
          <input type="checkbox" onChange={onCheckAll} checked={isAllChecked()} />
        </th>,
        ExperimentViewUtil.getHeaderCell("start_time", <span>{"Date"}</span>, true, onSortBy, sortState),
        ExperimentViewUtil.getHeaderCell("user_id", <span>{"User"}</span>, true, onSortBy, sortState),
        ExperimentViewUtil.getHeaderCell("source", <span>{"Source"}</span>, true, onSortBy, sortState),
        ExperimentViewUtil.getHeaderCell("source_version", <span>{"Version"}</span>, true, onSortBy, sortState)
      ];
      return (
        <Table hover>
        <colgroup span="7"/>
        <colgroup span="1"/>
        <colgroup span="1"/>
        <tbody>
        <tr>
            {columns}
            <th className="top-row left-border" scope="colgroup"
                colSpan="1">Parameters
            </th>
            <th className="top-row left-border" scope="colgroup"
                colSpan="1">Metrics
            </th>
            <tr>

            </tr>
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
        const sortedClassName = (isMetric, isParam, key) => {
            if (sortState.isMetric !== isMetric
                || sortState.isParam !== isParam
                || sortState.key !== key) {
                return "sortable";
            }
            return "sortable sorted " + (sortState.ascending ? "asc" : "desc");
        };
        const getHeaderCell = (key, text, sortable) => {
            let onClick = () => {};
            if (sortable) {
                onClick = () => onSortBy(false, false, key);
            }
            return <th key={"meta-" + key} className={"bottom-row " + sortedClassName(false, false, key)}
                       onClick={onClick}>{text}</th>;
        };

        const numParams = paramKeyList.length;
        const numMetrics = metricKeyList.length;
        const columns = [
            <th key="meta-check" className="bottom-row">
                <input type="checkbox" onChange={onCheckAll} checked={isAllCheckedBool} />
            </th>,
            getHeaderCell("start_time", <span>{"Date"}</span>, true),
            getHeaderCell("user_id", <span>{"User"}</span>, true),
            getHeaderCell("source", <span>{"Source"}</span>, true),
            getHeaderCell("source_version", <span>{"Version"}</span>, true)
        ];
        paramKeyList.forEach((paramKey, i) => {
            const className = "bottom-row "
                + (i === 0 ? "left-border " : "")
                + sortedClassName(false, true, paramKey);
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
                + sortedClassName(true, false, metricKey);
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

// export default connect(mapStateToProps)(ExperimentRunsTableNew);
export default ExperimentRunsTableNew;

