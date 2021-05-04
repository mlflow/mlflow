import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import debounce from 'lodash/debounce';

import { GenericInputModal } from './GenericInputModal';
import { ExperimentIdForm, NEW_ID_FIELD } from './ExperimentIdForm';
import { getExperimentIdValidator } from '../../../common/forms/validations';

import { moveRunApi, openErrorModal } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';

export class MoveRunModalImpl extends Component {
  // constructor(props) {
  //   super(props);
  //   this.handleSubmit = this.handleSubmit.bind(this);
  // }

  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    experimentId: PropTypes.string,
    experimentName: PropTypes.string,
    selectedRunIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    openErrorModal: PropTypes.func.isRequired,
    moveRunApi: PropTypes.func.isRequired,
  };

  handleSubmit = (values) => {
    const movePromises = [];
    const newExperimentId = values[NEW_ID_FIELD];
    this.props.selectedRunIds.forEach((runId) => {
      console.log(runId, this.props.experimentId, newExperimentId);
      movePromises.push(this.props.moveRunApi(
        runId, this.props.experimentId, newExperimentId
      ));
    });
    return Promise.all(movePromises).catch((err) => {
      this.props.openErrorModal('While moving an experiment run, an error occurred.');
    });
  };

  debouncedExperimentIdValidator = debounce(
    getExperimentIdValidator(() => this.props.experimentId, () => this.props.experimentIds),
    400,
  );

  render() {
    const { isOpen, experimentId } = this.props;
    return (
      <GenericInputModal
        title='Move to Experiment'
        okText='Move'
        isOpen={isOpen}
        handleSubmit={this.handleSubmit}
        onClose={this.props.onClose}
      >
        <ExperimentIdForm
          name='Enter experiment ID'
          visible={isOpen}
          validator={this.debouncedExperimentIdValidator}
        />
      </GenericInputModal>
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  const experimentIds = experiments.map((e) => e.getExperimentId());
  return { experimentIds };
};

const mapDispatchToProps = {
  moveRunApi,
  openErrorModal,
};

export const MoveRunModal = connect(mapStateToProps, mapDispatchToProps)(MoveRunModalImpl);


