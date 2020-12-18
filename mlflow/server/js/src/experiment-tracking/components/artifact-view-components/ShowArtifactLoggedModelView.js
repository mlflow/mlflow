import React, { Component } from 'react';
import PropTypes from 'prop-types';
import yaml from 'js-yaml';
import { MLMODEL_FILE_NAME } from '../../constants';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { SchemaTable } from '../../../model-registry/components/SchemaTable';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { RegisteringModelDocUrl, ModelSignatureUrl } from '../../../common/constants';

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

  renderCodeSnippet() {
    const { artifactRootUri, path } = this.props;
    const modelPath = `${artifactRootUri}/${path}`;
    return (
      <>
        <div className='content'>Predict on a Spark DataFrame:</div>
        <SyntaxHighlighter
          language={'python'}
          style={style}
          customStyle={styles.codeContent}
          wrapLines
          lineProps={{ style: { wordBreak: 'break-all', whiteSpace: 'pre-wrap' } }}
          wrapLongLines={false}
        >
          {`import mlflow\n` +
            `logged_model = '${modelPath}'\n\n` +
            `# Load model as a Spark UDF.\n` +
            `loaded_model = mlflow.pyfunc.spark_udf(logged_model)\n\n` +
            `# Predict on a Spark DataFrame.\n` +
            `df.withColumn(loaded_model, 'my_predictions')`}
        </SyntaxHighlighter>
        <div className='content'>Predict on a Pandas DataFrame:</div>
        <SyntaxHighlighter
          language={'python'}
          style={style}
          customStyle={styles.codeContent}
          wrapLines
          lineProps={{ style: { wordBreak: 'break-all', whiteSpace: 'pre-wrap' } }}
          wrapLongLines={false}
        >
          {`import mlflow\n` +
            `logged_model = '${modelPath}'\n\n` +
            `# Load model as a PyFuncModel.\n` +
            `loaded_model = mlflow.pyfunc.load_model(logged_model)\n\n` +
            `# Predict on a Pandas DataFrame.\n` +
            `import pandas as pd\n` +
            `loaded_model.predict(pd.DataFrame(data))`}
        </SyntaxHighlighter>
      </>
    );
  }
  static getLearnModelRegistryLinkUrl = () => RegisteringModelDocUrl;

  render() {
    if (this.state.loading) {
      return <div className='artifact-logged-model-view-loading'>Loading...</div>;
    } else if (this.state.error) {
      return (
        <div className='artifact-logged-model-view-error'>
          Couldn't load model information due to an error.
        </div>
      );
    } else {
      return (
        <div className='ShowArtifactPage'>
          <div className='artifact-logged-model-view-header' style={styles.header}>
            <h1>MLflow Model</h1>
            The code snippets below demonstrate how to make predictions using the logged model.{' '}
            {this.props.registeredModelLink ? (
              <>
                This model is also registered to the{' '}
                <a
                  href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()}
                  target='_blank'
                >
                  model registry
                </a>
                .
              </>
            ) : (
              <>
                You can also{' '}
                <a
                  href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()}
                  target='_blank'
                >
                  register it to the model registry
                </a>
                .
              </>
            )}
          </div>
          <hr />
          <div className='artifact-logged-model-view-schema-table' style={styles.schema}>
            <h2 style={styles.label}>Model schema</h2>
            <div className='content'>
              Input and output schema for your model.{' '}
              <a href={ModelSignatureUrl} target='_blank'>
                Learn more
              </a>
            </div>
            <div style={styles.schemaContent}>
              <SchemaTable
                schema={{ inputs: this.state.inputs, outputs: this.state.outputs }}
                defaultExpandAllRows
              />
            </div>
          </div>
          <div className='artifact-logged-model-view-code-group' style={styles.codeGroup}>
            <h2 style={styles.label}>Make Predictions</h2>
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
            inputs: JSON.parse(parsedJson.signature.inputs || '{}'),
            outputs: JSON.parse(parsedJson.signature.outputs || '{}'),
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
  label: {
    fontSize: 17,
    marginBottom: 16,
  },
  header: {
    marginTop: 16,
    marginBottom: 16,
    marginLeft: 16,
  },
  schema: {
    width: '35%',
    marginLeft: 16,
    float: 'left',
  },
  codeGroup: {
    width: '60%',
    marginRight: 16,
    float: 'right',
  },
  codeContent: {
    marginTop: 12,
  },
  schemaContent: {
    marginTop: 12,
  },
};

export default ShowArtifactLoggedModelView;
