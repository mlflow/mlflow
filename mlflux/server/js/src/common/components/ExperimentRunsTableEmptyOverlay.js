import React, { Component } from 'react';
import { css } from 'emotion';
import { LoggingRunsDocUrl } from '../constants';

/**
 * Display a helper overlay to on board new users on MLflow when the experiments runs table is
 * empty.
 */
export class ExperimentRunsTableEmptyOverlay extends Component {
  getLearnMoreLinkUrl = () => LoggingRunsDocUrl;

  render() {
    const learnMoreLinkUrl = this.getLearnMoreLinkUrl();
    return (
      <div className={`experiment-runs-table-empty-overlay ${classNames.wrapper}`}>
        <span>
          No runs yet.{' '}
          <a target='_blank' href={learnMoreLinkUrl}>
            Learn more
          </a>{' '}
          about how to create ML model training <br /> runs in this experiment.
        </span>
      </div>
    );
  }
}

const classNames = {
  wrapper: css({
    position: 'relative',
    top: '10px',
    fontSize: '13px',
    padding: '30px',
    width: '100 %',
    pointerEvents: 'all',
  }),
};
