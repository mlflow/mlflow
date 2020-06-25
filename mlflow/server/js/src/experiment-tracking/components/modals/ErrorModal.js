import React, { Component } from 'react';
import PropTypes from 'prop-types';
import ReactModal from 'react-modal';
import { Button, Modal } from 'react-bootstrap';
import { connect } from 'react-redux';
import { getErrorModalText, isErrorModalOpen } from '../../reducers/Reducers';
import { closeErrorModal } from '../../actions';

const modalStyles = {
  content: {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)',
    minWidth: 300,
    maxWidth: 600,
    padding: 0,
  },
  overlay: {
    backgroundColor: 'rgba(33, 37, 41, .75)',
  },
};

export class ErrorModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    text: PropTypes.string.isRequired,
  };

  render() {
    return (
      <ReactModal
        isOpen={this.props.isOpen}
        style={modalStyles}
        closeTimeoutMS={200}
        appElement={document.body}
        onRequestClose={this.props.onClose}
      >
        <Modal.Header>
          <Modal.Title>Oops!</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p style={{ marginBottom: '10px' }}>{this.props.text}</p>
        </Modal.Body>
        <Modal.Footer>
          <Button bsStyle='default' onClick={this.props.onClose} className='mlflow-form-button'>
            Close
          </Button>
        </Modal.Footer>
      </ReactModal>
    );
  }
}

const mapStateToProps = (state) => {
  const isOpen = isErrorModalOpen(state);
  const text = getErrorModalText(state);
  return {
    isOpen,
    text,
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    onClose: () => {
      dispatch(closeErrorModal());
    },
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(ErrorModalImpl);
