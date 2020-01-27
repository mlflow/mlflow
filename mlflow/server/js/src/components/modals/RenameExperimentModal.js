import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import RenameFormView from './RenameFormView';

import ReactModal from 'react-modal';

import { updateExperimentApi } from '../../Actions';


const modalStyles = {
  content: {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)',
    padding: 0,
  },
  overlay: {
    backgroundColor: 'rgba(33, 37, 41, .75)',
  },
};

export class RenameExperimentModal extends Component {
  state = {
    isSubmittingState: false,
  };

  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number,
    experimentName: PropTypes.string,
    onClose: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  /**
   * Form-submission handler with method signature as prescribed by Formik.
   * See https://github.com/jaredpalmer/formik#how-form-submission-works for an explanation
   * of how / when this method is called.
   */
  handleSubmit = (values, {setSubmitting, setStatus}) => {
    const { newExperimentName } = values;
    this.setState({ isSubmittingState: true });

    return this.props.dispatch(
      updateExperimentApi(this.props.experimentId, newExperimentName)).then(() => {
        this.setState({ isSubmittingState: false });
        setSubmitting(false);
        this.onRequestCloseHandler();
      }).catch((err) => {
        this.setState({ isSubmittingState: false });
        setSubmitting(false);
        // TODO: ??
        setStatus({errorMsg: err.getUserVisibleError()});
      });
  };


  onRequestCloseHandler = () => {
    if (!this.state.isSubmittingState) {
      this.props.onClose();
    }
  }

  render() {
    const { isOpen, experimentId, experimentName } = this.props;
    return (
    <ReactModal
      isOpen={isOpen}
      onRequestClose={this.onRequestCloseHandler}
      style={modalStyles}
      closeTimeoutMS={200}
      appElement={document.body}
    >
      <a className="modal-close-link">
        <i onClick={this.onRequestCloseHandler} className="fas fa-times"/>
      </a>
      <RenameFormView
        onSubmit={this.handleSubmit}
        onClose={this.onRequestCloseHandler}
        name={experimentName}
        experimentId={experimentId}
        type={"experiment"}
      />
    </ReactModal>);
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(RenameExperimentModal);
