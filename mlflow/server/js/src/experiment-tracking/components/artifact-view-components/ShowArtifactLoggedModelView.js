import React, { Component } from 'react';
import PropTypes from 'prop-types';
import yaml from 'js-yaml';
import '../../../common/styles/CodeSnippet.css';
import { MLMODEL_FILE_NAME } from '../../constants';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { SchemaTable } from '../../../model-registry/components/SchemaTable';
import { RegisteringModelDocUrl, ModelSignatureUrl } from '../../../common/constants';
import { Typography } from 'antd';
import { FormattedMessage } from 'react-intl';

const { Paragraph } = Typography;

class ShowArtifactLoggedModelView extends Component {
  constructor(props) {
    super(props);
    this.fetchLoggedModelMetadata = this.fetchLoggedModelMetadata.bind(this);
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    getArtifact: PropTypes.func,
    artifactRootUri: PropTypes.string.isRequired,
    registeredModelLink: PropTypes.string,
  };

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    inputs: undefined,
    outputs: undefined,
  };

  componentDidMount() {
    this.fetchLoggedModelMetadata();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchLoggedModelMetadata();
    }
  }
  static getLearnModelRegistryLinkUrl = () => RegisteringModelDocUrl;

  renderModelRegistryText() {
    return this.props.registeredModelLink ? (
      <>
        <FormattedMessage
          defaultMessage='This model is also registered to the <link>model registry</link>.'
          description='Sub text to tell the users where the registered models are located '
          values={{
            link: (chunks) => (
              <a href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()} target='_blank'>
                {chunks}
              </a>
            ),
          }}
        />
      </>
    ) : (
      <>
        <FormattedMessage
          // eslint-disable-next-line max-len
          defaultMessage='You can also <link>register it to the model registry</link> to version control'
          description='Sub text to tell the users where one can go to register the model artifact'
          values={{
            link: (chunks) => (
              <a href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()} target='_blank'>
                {chunks}
              </a>
            ),
          }}
        />
      </>
    );
  }

  sparkDataFrameCodeText(modelPath) {
    return (
      `import mlflow\n` +
      `logged_model = '${modelPath}'\n\n` +
      `# Load model as a Spark UDF.\n` +
      `loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)\n\n` +
      `# Predict on a Spark DataFrame.\n` +
      `df.withColumn('predictions', loaded_model(*column_names)).collect()`
    );
  }

  pandasDataFrameCodeText(modelPath) {
    return (
      `import mlflow\n` +
      `logged_model = '${modelPath}'\n\n` +
      `# Load model as a PyFuncModel.\n` +
      `loaded_model = mlflow.pyfunc.load_model(logged_model)\n\n` +
      `# Predict on a Pandas DataFrame.\n` +
      `import pandas as pd\n` +
      `loaded_model.predict(pd.DataFrame(data))`
    );
  }

  renderCodeSnippet() {
    const { runUuid, path } = this.props;
    const modelPath = `runs:/${runUuid}/${path}`;
    return (
      <>
        <div className='content' style={styles.item}>
          <h3 style={styles.itemHeader}>
            <FormattedMessage
              defaultMessage='Predict on a Spark DataFrame:'
              // eslint-disable-next-line max-len
              description='Section heading to display the code block on how we can use registered model to predict using spark DataFrame'
            />
          </h3>
          <Paragraph copyable={{ text: this.sparkDataFrameCodeText(modelPath) }}>
            <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap', marginTop: 10 }}>
              <div className='code'>
                <span className='code-keyword'>import</span> mlflow{`\n`}
                logged_model = <span className='code-string'>{`'${modelPath}'`}</span>
              </div>
              <br />
              <div className='code'>
                <span className='code-comment'>
                  {'# '}
                  <FormattedMessage
                    defaultMessage='Load model as a Spark UDF.'
                    description='Code comment which states how to load model using spark UDF'
                  />
                </span>
                {`\n`}
                loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
              </div>
              <br />
              <div className='code'>
                <span className='code-comment'>
                  {'# '}
                  <FormattedMessage
                    defaultMessage='Predict on a Spark DataFrame.'
                    // eslint-disable-next-line max-len
                    description='Code comment which states on how we can predict using spark DataFrame'
                  />
                </span>
                {`\n`}
                df.withColumn(<span className='code-string'>'predictions'</span>,
                loaded_model(*columns)).collect()
              </div>
            </pre>
          </Paragraph>
        </div>
        <div className='content' style={styles.item}>
          <h3 style={styles.itemHeader}>
            <FormattedMessage
              defaultMessage='Predict on a Pandas DataFrame:' // eslint-disable-next-line max-len
              description='Section heading to display the code block on how we can use registered model to predict using pandas DataFrame'
            />
          </h3>
          <Paragraph copyable={{ text: this.pandasDataFrameCodeText(modelPath) }}>
            <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
              <div className='code'>
                <span className='code-keyword'>import</span> mlflow{`\n`}
                logged_model = <span className='code-string'>{`'${modelPath}'`}</span>
              </div>
              <br />
              <div className='code'>
                <span className='code-comment'>
                  {'# '}
                  <FormattedMessage
                    defaultMessage='Load model as a PyFuncModel.'
                    description='Code comment which states how to load model using PyFuncModel'
                  />
                </span>
                {`\n`}
                loaded_model = mlflow.pyfunc.load_model(logged_model)
              </div>
              <br />
              <div className='code'>
                <span className='code-comment'>
                  {'# '}
                  <FormattedMessage
                    defaultMessage='Predict on a Pandas DataFrame.'
                    // eslint-disable-next-line max-len
                    description='Code comment which states on how we can predict using pandas DataFrame'
                  />
                </span>
                {`\n`}
                <span className='code-keyword'>import</span> pandas{' '}
                <span className='code-keyword'>as</span> pd{`\n`}
                loaded_model.predict(pd.DataFrame(data))
              </div>
            </pre>
          </Paragraph>
        </div>
      </>
    );
  }

  render() {
    if (this.state.loading) {
      return (
        <div className='artifact-logged-model-view-loading'>
          <FormattedMessage
            defaultMessage='Loading...'
            description='Loading state text for the artifact model view'
          />
        </div>
      );
    } else if (this.state.error) {
      return (
        <div className='artifact-logged-model-view-error'>
          <FormattedMessage
            defaultMessage="Couldn't load model information due to an error."
            description='Error state text when the model artifact was unable to load'
          />
        </div>
      );
    } else {
      return (
        <div className='ShowArtifactPage'>
          <div
            className='artifact-logged-model-view-header'
            style={{ marginTop: 16, marginBottom: 16, marginLeft: 16 }}
          >
            <h1>
              <FormattedMessage
                defaultMessage='MLflow Model'
                description='Heading text for mlflow model artifact'
              />
            </h1>
            <FormattedMessage
              // eslint-disable-next-line max-len
              defaultMessage='The code snippets below demonstrate how to make predictions using the logged model.'
              // eslint-disable-next-line max-len
              description='Subtext heading explaining the below section of the model artifact view on how users can prediction using the registered logged model'
            />{' '}
            {this.renderModelRegistryText()}
          </div>
          <hr />
          <div
            className='artifact-logged-model-view-schema-table'
            style={{ width: '35%', marginLeft: 16, float: 'left' }}
          >
            <h2 style={styles.columnLabel}>
              <FormattedMessage
                defaultMessage='Model schema'
                // eslint-disable-next-line max-len
                description='Heading text for the model schema of the registered model from the experiment run'
              />
            </h2>
            <div className='content'>
              <h3 style={styles.itemHeader}>
                <FormattedMessage
                  defaultMessage='Input and output schema for your model. <link>Learn more</link>'
                  // eslint-disable-next-line max-len
                  description='Input and output params of the model that is registered from the experiment run'
                  values={{
                    link: (chunks) => (
                      <a href={ModelSignatureUrl} target='_blank'>
                        {chunks}
                      </a>
                    ),
                  }}
                />
              </h3>
            </div>
            <div style={{ marginTop: 12 }}>
              <SchemaTable
                schema={{ inputs: this.state.inputs, outputs: this.state.outputs }}
                defaultExpandAllRows
              />
            </div>
          </div>
          <div
            className='artifact-logged-model-view-code-group'
            style={{ width: '60%', marginRight: 16, float: 'right' }}
          >
            <h2 style={styles.columnLabel}>
              <FormattedMessage
                defaultMessage='Make Predictions'
                // eslint-disable-next-line max-len
                description='Heading text for the prediction section on the registered model from the experiment run'
              />
            </h2>
            <div className='artifact-logged-model-view-code-content'>
              {this.renderCodeSnippet()}
            </div>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchLoggedModelMetadata() {
    const modelFileLocation = getSrc(`${this.props.path}/${MLMODEL_FILE_NAME}`, this.props.runUuid);
    this.props
      .getArtifact(modelFileLocation)
      .then((response) => {
        const parsedJson = yaml.load(response);
        if (parsedJson.signature) {
          this.setState({
            inputs: JSON.parse(parsedJson.signature.inputs || '[]'),
            outputs: JSON.parse(parsedJson.signature.outputs || '[]'),
          });
        } else {
          this.setState({ inputs: '', outputs: '' });
        }
        this.setState({ loading: false });
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }
}

const styles = {
  columnLabel: {
    fontSize: 18,
    marginBottom: 16,
  },
  item: {
    position: 'relative',
  },
  itemHeader: {
    fontSize: 15,
  },
};

export default ShowArtifactLoggedModelView;
