import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Button, Modal } from 'react-bootstrap';
import { withRouter } from 'react-router-dom';
import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
import TextField from '../fields/TextField';

/**
 * Component that renders a form for updating a run's name. Expects to be 'closeable'
 * (i.e. rendered within a closeable dialog) and so accepts an `onClose` callback.
 */
class RenameRunFormView extends Component {
  static propTypes = {
    onSubmit: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
    runName: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired,
  }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
      status,
    } = renderProps;
    const saveText = isSubmitting ? "Saving..." : "Save";
    const handleFocus = (event) => {
      event.target.select();
    };
    return <form onSubmit={handleSubmit} className="rename-run-form">
      <Modal.Header>
        <Modal.Title>
          Rename Run
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Field
            name="newRunName"
            label="New run name:"
            autoFocus
            onFocus={handleFocus}
            autoComplete="off"
            component={TextField}
        />
        { status && status.errorMsg &&
          <div className="text-danger">
            <i className="fas fa-exclamation-triangle"></i> {status.errorMsg}
          </div>
        }
      </Modal.Body>
      <Modal.Footer>
        <Button
          bsStyle="default"
          disabled={isSubmitting}
          onClick={this.props.onClose}
          className="mlflow-form-button"
        >
          Cancel
        </Button>
        <Button
          bsStyle="primary"
          type="submit"
          disabled={isSubmitting}
          className="mlflow-save-button mlflow-form-button"
        >
          {saveText}
        </Button>
      </Modal.Footer>
    </form>;
  }

  render() {
    return (<div>
      <Formik
        initialValues={{newRunName: this.props.runName}}
        validationSchema={validationSchema}
        onSubmit={this.props.onSubmit}
        render={this.renderForm}/>
      </div>);
  }
}

export default withRouter(RenameRunFormView);
