import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Button } from 'react-bootstrap';
import { withRouter } from 'react-router-dom';
import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
import TextField from './TextField';

/**
 * Component that renders a form for updating a run's name. Expects to be 'closeable'
 * (i.e. rendered within a closeable dialog) and so accepts an `onClose` callback.
 */
class RenameRunFormView extends Component {
  static propTypes = {
    onSubmit: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
    runName: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired
  }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
    } = renderProps;
    const saveText = isSubmitting ? "Saving..." : "Save";
    return <form onSubmit={handleSubmit} style={styles.form}>
      <div><h2 style={{"marginTop": 0}}>Rename Run</h2></div>
      <div style={{...styles.formField, "width": "100%"}}>
        <Field
            name="newRunName"
            label="New run name:"
            component={TextField}
            autoFocus
        />
      </div>
      <div style={styles.buttonsDiv}>
        <Button
          bsStyle="default"
          className="cancel-button"
          disabled={isSubmitting}
          onClick={this.props.onClose}
        >
          Cancel
        </Button>
        <Button bsStyle="primary" type="submit" className="save-button" disabled={isSubmitting}>
          {saveText}
        </Button>
      </div>
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

const styles = {
  form: {
    width: 480,
  },
  buttonsDiv: {
    display: "flex",
    justifyContent: "flex-end",
  },
  formField: {
    marginBottom: 16,
  },
  label: {
    marginBottom: 4,
  },
};
