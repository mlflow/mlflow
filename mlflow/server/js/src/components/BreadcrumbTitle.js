import React, { Component } from "react";
import PropTypes from "prop-types";
import { Experiment } from "../sdk/MlflowMessages";
import { Link } from 'react-router-dom';
import Routes from "../Routes";
import Utils from '../utils/Utils';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';

/**
 * A title component that creates a <h1> with breadcrumbs pointing to an experiment and optionally
 * a run or a run comparison page.
 */

const DROPDOWN_MENU = 'dropdownMenu';


class BreadcrumbTitle extends Component {
  constructor(props) {
    super(props);
    // TODO do we need this here and below?
    this.state.showMenu = true;
  }
  // title={Utils.getRunDisplayName(this.props.tags, run.getRunUuid())}
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String), // Optional because not all pages are nested under runs
    // TODO this might need to be an array for multiple runs?
    tags: PropTypes.object,
  };

  state = {
      showMenu: true,
      above: false,
      dropdownHeight: 0,
  };

  /**
   * Hide the dropdown menu for a table item.
   **/
  hideMenu() {
    this.setState({
      showMenu: true,
      above: false,
      dropdownHeight: 0,
    });
  }

  renameRun() {
    console.log("Hi! In renameRun() in BreadCrumbTitle.js");
  }

  render() {
    const {experiment, runUuids, title} = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link to={Routes.getExperimentPageRoute(experimentId)}>
        {experiment.getName()}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink = (runUuids.length === 1 ?
        <div>
          <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])} key="link">
            {Utils.getRunDisplayName(this.props.tags, runUuids[0])}
          </Link>
        </div>
        :
        <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)} key="link">
          Comparing {runUuids.length} Runs
        </Link>
      );
    }
    const chevron = <i className="fas fa-chevron-right breadcrumb-chevron" key="chevron"/>;
    return (
      <h1>
        {experimentLink}
        {chevron}
        { runsLink ? [runsLink, chevron] : [] }
        {title}
      </h1>
    );
  }
}


const mapStateToProps = (state, ownProps) => {
  const { experimentId, runUuids } = ownProps;
  const experiment = getExperiment(experimentId, state);
  // TODO handle array
  const tags = getRunTags(runUuids[0], state);
  const run = getRunInfo(runUuids[0], state);
  return { run, experiment, tags };
};


export default connect(mapStateToProps)(BreadcrumbTitle);
