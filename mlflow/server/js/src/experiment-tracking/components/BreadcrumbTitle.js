import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Experiment } from '../sdk/MlflowMessages';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import './BreadcrumbTitle.css';

/**
 * A title component that creates a <h1> with breadcrumbs pointing to an experiment and optionally
 * a run or a run comparison page.
 */
export class BreadcrumbTitle extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    // Optional because not all pages are nested under runs
    runUuids: PropTypes.arrayOf(PropTypes.string),
    runNames: PropTypes.arrayOf(PropTypes.string),
    title: PropTypes.any.isRequired,
  };

  render() {
    const { experiment, runUuids, runNames, title } = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link
        to={Routes.getExperimentPageRoute(experimentId)}
        className='truncate-text single-line breadcrumb-title'
      >
        {experiment.getName()}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink =
        runUuids.length === 1 ? (
          <Link
            to={Routes.getRunPageRoute(experimentId, runUuids[0])}
            key='link'
            className='truncate-text single-line breadcrumb-title'
          >
            {runNames[0]}
          </Link>
        ) : (
          <Link
            to={Routes.getCompareRunPageRoute(runUuids, experimentId)}
            key='link'
            className='truncate-text single-line breadcrumb-title'
          >
            Comparing {runUuids.length} Runs
          </Link>
        );
    }
    const chevron = <i className='fas fa-chevron-right breadcrumb-chevron' key='chevron' />;
    return (
      <h1 className='breadcrumb-header'>
        {experimentLink}
        {chevron}
        {runsLink ? [runsLink, chevron] : []}
        <span className='truncate-text single-line breadcrumb-title'>{title}</span>
      </h1>
    );
  }
}
