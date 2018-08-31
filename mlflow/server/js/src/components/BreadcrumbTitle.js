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
export default class BreadcrumbTitle extends Component {
  constructor(props) {
    super(props);
  }
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String), // Optional because not all pages are nested under runs
    title: PropTypes.string, // Optional. A title is inferred
  };

  render() {
    const {experiment, runUuids, title, runName} = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link to={Routes.getExperimentPageRoute(experimentId)}>
        {experiment.getName()}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink = (runUuids.length === 1 ?
        <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])} key="link">
          {runName}
        </Link>
        :
        <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)} key="link">
          Comparing {runUuids.length} Runs
        </Link>
      );
    }
    const chevron = <i className="fas fa-chevron-right breadcrumb-chevron" key="chevron"/>;
    return (
      <h1 style={{display: "inline"}}>
        {experimentLink}
        {chevron}
        { runsLink ? [runsLink] : [] }
        { title ? [chevron, title] : []}
      </h1>
    );
  }
}
