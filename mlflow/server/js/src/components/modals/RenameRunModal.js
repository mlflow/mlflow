import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import RenameRunFormView from './RenameRunFormView';

import Utils from '../../utils/Utils';
import ReactModal from 'react-modal';
import { getRunTags } from '../../reducers/Reducers';

import { setTagApi, getUUID } from '../../Actions';
import { withRouter } from 'react-router-dom';
import Routes from "../../Routes";


const modalStyles = {
  content : {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)'
  },
  overlay: {
    backgroundColor: 'rgba(33, 37, 41, .75)'
  }
};

class RenameRunModal extends Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.onRequestCloseHandler = this.onRequestCloseHandler.bind(this);
  }

  state = {
    isSubmittingState: false,
  }

  static propTypes = {
    open: PropTypes.bool,
    experimentId: PropTypes.number.isRequired,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
  }

  /**
   * Form-submission handler with method signature as prescribed by Formik.
   * See https://github.com/jaredpalmer/formik#how-form-submission-works for an explanation
   * of how / when this method is called.
   */
  handleSubmit = (
    values,
    {
      props,
      setSubmitting,
      setErrors /* setValues, setStatus, and other goodies */,
    }) => {
      const { newRunName } = values;
      this.setState({isSubmittingState: true});
      const tagKey = Utils.runNameTag;
      const setTagRequestId = getUUID();
      return this.props.dispatch(
        setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId)).catch((err) => {
        // TODO: remove alert, redirect to an error page on failed requests once one exists
        alert("Unable to rename run, got error '" + err + "'. Redirecting to parent experiment " +
          "page.");
        this.props.history.push(Routes.getExperimentPageRoute(this.props.experimentId));
      }).finally(() => {
        this.setState({isSubmittingState: false});
        setSubmitting(false);
        this.onRequestCloseHandler();
      })
    }


  renderForm() {
    const { runName, experimentId } = this.props;
    return <RenameRunFormView
      onSubmit={this.handleSubmit}
      onClose={this.onRequestCloseHandler}
      runName={runName}
      experimentId={experimentId}/>
  }

  onRequestCloseHandler(event) {
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
      appElement={document.body}
    >
      {this.renderForm()}
      <a className="exit-link">
        <i onClick={this.onRequestCloseHandler} className="fas fa-times"/>
      </a>
    </ReactModal>);
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(function() { return {} }, mapDispatchToProps)(withRouter(RenameRunModal))
