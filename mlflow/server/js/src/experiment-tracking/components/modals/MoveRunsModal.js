import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl } from 'react-intl';

import { GenericInputModal } from './GenericInputModal';
import { MoveRunsForm } from './MoveRunsForm';

import { moveRunsApi } from '../../actions';

export class MoveRunsModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    moveRunsApi: PropTypes.func.isRequired,
    experimentList: PropTypes.array.isRequired,
    selectedRunIds: PropTypes.array.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
    onSuccess: PropTypes.func,
  };
  constructor() {
    super();
    this.form = React.createRef();
  }

  handleMoveRuns = (values) => {
    return this.props
      .moveRunsApi(this.props.selectedRunIds, values.experimentId)
      .then(() => this.props.onSuccess?.());
  };

  render() {
    const { isOpen } = this.props;
    return (
      <GenericInputModal
        title={this.props.intl.formatMessage({
          defaultMessage: 'Move Runs',
          description: 'Modal title to move runs to another experiment',
        })}
        okText={this.props.intl.formatMessage({
          defaultMessage: 'Move',
          description: 'Modal button text to move runs to another experiment',
        })}
        isOpen={isOpen}
        handleSubmit={this.handleMoveRuns}
        onClose={this.props.onClose}
      >
        <MoveRunsForm
          visible={isOpen}
          experimentList={this.props.experimentList}
        />
      </GenericInputModal>
    );
  }
}

const mapDispatchToProps = {
  moveRunsApi,
};

export const MoveRunsModalWithIntl = injectIntl(MoveRunsModalImpl);
export const MoveRunsModal = connect(undefined, mapDispatchToProps)(MoveRunsModalWithIntl);
