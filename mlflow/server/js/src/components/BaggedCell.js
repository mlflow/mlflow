import Utils from "../utils/Utils";
import React, { Component, PureComponent } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { Dropdown, MenuItem } from 'react-bootstrap';
import {RunInfo} from "../sdk/MlflowMessages";
import classNames from 'classnames';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';

const styles = {
  sortArrow: {
    marginLeft: "2px",
  },
  sortContainer: {
    minHeight: "18px",
  },
  sortToggle: {
    cursor: "pointer",
  },
  sortKeyName: {
    display: "inline-block"
  },
  metricParamCellContent: {
    display: "inline-block",
    maxWidth: 120,
  },
  metricParamNameContainer: {
    verticalAlign: "middle",
  },
};

export default class BaggedCell extends Component {

  constructor(props) {
    super(props);
    this.showDropdownHandler = this.showDropdownHandler.bind(this);
  }

  static propTypes = {
    keyName: PropTypes.string.isRequired,
    value: PropTypes.string.isRequired,
    // TODO sortIcon

    // (key) => Unit (sets hover state to that key, other args expected to be prefilled)
    onHover: PropTypes.func.isRequired,
    setSortByHandler: PropTypes.func.isRequired,
    isParam: PropTypes.bool.isRequired,
    isMetric: PropTypes.bool.isRequired,
    isHovered: PropTypes.bool.isRequired,
    onRemoveBagged: PropTypes.func.isRequired,

  };

  state = {
    showDropdown: false,
  };

  showDropdownHandler() {
    this.setState({showDropdown: true});
  }

  shouldComponentUpdate(nextProps, nextState) {
    return this.props.isHovered !== nextProps.isHovered;
  }

  getDropdown() {
    const { keyName, value, onHover, setSortByHandler, isParam, isMetric, isHovered, onRemoveBagged} = this.props;
    return ( <Dropdown id="dropdown-custom-1" style={{width: 250}} open={true}>
        <ExperimentRunsSortToggle
          bsRole="toggle"
          className={"metric-param-sort-toggle"}
          onClick={() => this.setState({showDropdown: false})}
        >
              <span
                className="run-table-container underline-on-hover"
                style={styles.metricParamCellContent}
                title={keyName}
              >
                {keyName}:
              </span>
        </ExperimentRunsSortToggle>
        <span
          className="metric-param-value run-table-container"
          style={styles.metricParamCellContent}
        >
              {value}
        </span>
        <Dropdown.Menu className="mlflow-menu">
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => setSortByHandler(isMetric, isParam, keyName, true)}
          >
            Sort ascending
          </MenuItem>
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => setSortByHandler(isMetric, isParam, keyName, false)}
          >
            Sort descending
          </MenuItem>
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => onRemoveBagged(isParam, keyName)}
          >
            Display in own column
          </MenuItem>
        </Dropdown.Menu>
      </Dropdown>);
  }

  getCellContents() {
    const { keyName, value, onHover, setSortByHandler, isParam, isMetric, isHovered, onRemoveBagged} = this.props;
    if (this.state.showDropdown) {
      return this.getDropdown();
    }
    return         ([<ExperimentRunsSortToggle
      bsRole="toggle"
      className={"metric-param-sort-toggle"}
      onClick={this.showDropdownHandler}
    >
              <span
                className="run-table-container underline-on-hover"
                style={styles.metricParamCellContent}
                title={keyName}
              >
                {keyName}:
              </span>
    </ExperimentRunsSortToggle>,         <span
      className="metric-param-value run-table-container"
      style={styles.metricParamCellContent}
    >
              {value}
        </span>]);
    // return (
    //   <span
    //     className="run-table-container underline-on-hover"
    //     style={styles.metricParamCellContent}
    //   >
    //               {keyName}:
    //     <span
    //       className="metric-param-value run-table-container"
    //       style={styles.metricParamCellContent}
    //     >
    //           {value}
    //         </span>
    //   </span>
    // )
  }


  render() {
    const { keyName, value, onHover, setSortByHandler, isParam, isMetric, isHovered, onRemoveBagged} = this.props;
    const cellClass = classNames("metric-param-content", "metric-param-cell");
    const contents = this.getCellContents();
    return (
      <span
        className={cellClass}
        onMouseEnter={() => onHover({isParam: isParam, isMetric: isMetric, key: keyName})}
        onMouseLeave={() => onHover({isParam: false, isMetric: false, key: ""})}
      >
          {contents}
      </span>
    );
  }
}