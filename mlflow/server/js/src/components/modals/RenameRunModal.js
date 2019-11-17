import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import RenameRunFormView from './RenameRunFormView';

import Utils from '../../utils/Utils';
import ReactModal from 'react-modal';

import { setTagApi, getUUID } from '../../Actions';


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

export class RenameRunModal extends Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.onRequestCloseHandler = this.onRequestCloseHandler.bind(this);
  }

  state = {
    isSubmittingState: false,
  };

  static propTypes = {
    open: PropTypes.bool,
    experimentId: PropTypes.number.isRequired,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  /**
   * Form-submission handler with method signature as prescribed by Formik.
   * See https://github.com/jaredpalmer/formik#how-form-submission-works for an explanation
   * of how / when this method is called.
   */
  handleSubmit = (
    values,
    {
      setSubmitting,
      setStatus,
    }) => {
    const { newRunName } = values;
    this.setState({ isSubmittingState: true });
    const tagKey = Utils.runNameTag;
    const setTagRequestId = getUUID();
    return this.props.dispatch(
      setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId)).then(() => {
        this.setState({ isSubmittingState: false });
        setSubmitting(false);
        this.onRequestCloseHandler();
      }).catch((err) => {
        this.setState({ isSubmittingState: false });
        setSubmitting(false);
        setStatus({errorMsg: err.getUserVisibleError()});
      });
  };


  renderForm() {
    const { runName, experimentId } = this.props;
    return (<RenameRunFormView
      onSubmit={this.handleSubmit}
      onClose={this.onRequestCloseHandler}
      runName={runName}
      experimentId={experimentId}/>);
  }

  onRequestCloseHandler() {
    if (!this.state.isSubmittingState) {
      this.props.onClose();
    }
  }

  render() {
    const { open } = this.props;
    return (
    <ReactModal
      isOpen={open}
      onRequestClose={this.onRequestCloseHandler}
      style={modalStyles}
      closeTimeoutMS={200}
      appElement={document.body}
    >
      <a className="modal-close-link">
        <i onClick={this.onRequestCloseHandler} className="fas fa-times"/>
      </a>
      {this.renderForm()}
    </ReactModal>);
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(RenameRunModal);
