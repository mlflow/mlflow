import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { Button } from 'react-bootstrap';
import { withRouter } from 'react-router-dom';
import Routes from "../../Routes";
import { Formik, Field } from 'formik';

import { validationSchema } from './validation';

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
    return <form onSubmit={handleSubmit} style={{"width": "480px"}}>
      <div style={styles.formField}> <h2 style={{"marginTop": "0px"}}> Rename Run </h2> </div>
      <div style={styles.formField}> New run name: </div>
      <div style={{"width": "100%", "marginBottom": "16px"}}>
        <Field
            type="newRunName"
            name="newRunName"
            label="New Run Name"
            autoFocus
            style={{"width": "100%"}}
        />
      </div>
      <div style={styles.buttonsDiv}>
        <Button bsStyle="primary" type="submit" className="save-button" disabled={isSubmitting}>
          Save
        </Button>
        <Button bsStyle="default" className="cancel-button" disabled={isSubmitting}
          onClick={this.props.onClose}>
          Cancel
        </Button>
      </div>
    </form>;
  }

  render() {
    return <div>
      <Formik
        initialValues={{newRunName: this.props.runName}}
        validationSchema={validationSchema}
        onSubmit={this.props.onSubmit}
        render={this.renderForm}/>
    </div>
  }
}

export default withRouter(RenameRunFormView);

const styles = {
  buttonsDiv: {
    display: "flex",
    justifyContent: "flex-end",
  },
  formField: {
    marginBottom: "16px",
  }
};
