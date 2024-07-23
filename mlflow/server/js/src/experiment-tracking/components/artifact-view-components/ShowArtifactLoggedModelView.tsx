/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import yaml from 'js-yaml';
import '../../../common/styles/CodeSnippet.css';
import { MLMODEL_FILE_NAME, SERVING_INPUT_FILE_NAME } from '../../constants';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { SchemaTable } from '../../../model-registry/components/SchemaTable';
import {
  RegisteringModelDocUrl,
  ModelSignatureUrl,
  PyfuncDocUrl,
  CustomPyfuncModelsDocUrl,
} from '../../../common/constants';
import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import './ShowArtifactLoggedModelView.css';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';

const { Paragraph, Text, Title } = Typography;

type OwnProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
  artifactRootUri: string;
  registeredModelLink?: string;
};

type State = any;

type Props = OwnProps & typeof ShowArtifactLoggedModelView.defaultProps;

class ShowArtifactLoggedModelView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchLoggedModelMetadata = this.fetchLoggedModelMetadata.bind(this);
    this.fetchServingInputExample = this.fetchServingInputExample.bind(this);
  }

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    inputs: undefined,
    outputs: undefined,
    flavor: undefined,
    loader_module: undefined,
    serving_input: undefined,
  };

  componentDidMount() {
    this.fetchLoggedModelMetadata();
    this.fetchServingInputExample();
  }

  componentDidUpdate(prevProps: Props) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchLoggedModelMetadata();
      this.fetchServingInputExample();
    }
  }
  static getLearnModelRegistryLinkUrl = () => RegisteringModelDocUrl;

  renderModelRegistryText() {
    return this.props.registeredModelLink ? (
      <>
        <FormattedMessage
          defaultMessage="This model is also registered to the <link>model registry</link>."
          description="Sub text to tell the users where the registered models are located "
          values={{
            link: (
              chunks: any, // Reported during ESLint upgrade
            ) => (
              // eslint-disable-next-line react/jsx-no-target-blank
              <a href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()} target="_blank">
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
          defaultMessage="You can also <link>register it to the model registry</link> to version control"
          description="Sub text to tell the users where one can go to register the model artifact"
          values={{
            link: (
              chunks: any, // Reported during ESLint upgrade
            ) => (
              // eslint-disable-next-line react/jsx-no-target-blank
              <a href={ShowArtifactLoggedModelView.getLearnModelRegistryLinkUrl()} target="_blank">
                {chunks}
              </a>
            ),
          }}
        />
      </>
    );
  }

  sparkDataFrameCodeText(modelPath: any) {
    return (
      `import mlflow\n` +
      `from pyspark.sql.functions import struct, col\n` +
      `logged_model = '${modelPath}'\n\n` +
      `# Load model as a Spark UDF.\n` +
      `loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)\n\n` +
      `# Predict on a Spark DataFrame.\n` +
      `df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))`
    );
  }

  loadModelCodeText(modelPath: any, flavor: any) {
    return (
      `import mlflow\n` +
      `logged_model = '${modelPath}'\n\n` +
      `# Load model.\n` +
      `loaded_model = mlflow.${flavor}.load_model(logged_model)\n`
    );
  }

  pandasDataFrameCodeText(modelPath: any) {
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

  mlflowSparkCodeText(modelPath: any) {
    return (
      `import mlflow\n` +
      `logged_model = '${modelPath}'\n\n` +
      `# Load model\n` +
      `loaded_model = mlflow.spark.load_model(logged_model)\n\n` +
      `# Perform inference via model.transform()\n` +
      `loaded_model.transform(data)`
    );
  }

  validateModelForServingText(modelPath: any, servingInput?: string) {
    if (servingInput) {
      return `from mlflow.models import validate_serving_input

model_uri = '${modelPath}'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """${servingInput}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)`;
    } else {
      return `from mlflow.models import validate_serving_input

model_uri = '${modelPath}'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)`;
    }
  }

  renderNonPyfuncCodeSnippet() {
    const { flavor } = this.state;
    const { runUuid, path } = this.props;
    const modelPath = `runs:/${runUuid}/${path}`;

    if (flavor === 'mleap') {
      // MLeap models can't be reloaded in Python.
      return <></>;
    }

    return (
      <>
        <Title level={3}>
          <FormattedMessage
            defaultMessage="Load the model"
            // eslint-disable-next-line max-len
            description="Heading text for stating how to load the model from the experiment run"
          />
        </Title>
        <div className="artifact-logged-model-view-code-content">
          {/* @ts-expect-error TS(2322): Type '{ position: string; pre: { margin: number; }... Remove this comment to see the full error message */}
          <div css={styles.item}>
            <Paragraph
              dangerouslySetAntdProps={{
                copyable: { text: this.loadModelCodeText(modelPath, flavor) },
              }}
            >
              <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
                <div className="code">
                  <span className="code-keyword">import</span> mlflow{`\n`}
                  logged_model = <span className="code-string">{`'${modelPath}'`}</span>
                </div>
                <br />
                <div className="code">
                  <span className="code-comment">
                    {'# '}
                    <FormattedMessage
                      defaultMessage="Load model"
                      description="Code comment which states how to load the model"
                    />
                  </span>
                  {`\n`}
                  loaded_model = mlflow.{flavor}.load_model(logged_model)
                </div>
                <br />
              </pre>
            </Paragraph>
            <FormattedMessage
              // eslint-disable-next-line max-len
              defaultMessage="See the documents below to learn how to customize this model and deploy it for batch or real-time scoring using the pyfunc model flavor."
              // eslint-disable-next-line max-len
              description="Subtext heading for a list of documents that describe how to customize the model using the mlflow.pyfunc module"
            />
            <ul>
              <li>
                <a href={PyfuncDocUrl}>API reference for the mlflow.pyfunc module</a>
              </li>
              <li>
                <a href={CustomPyfuncModelsDocUrl}>Creating custom Pyfunc models</a>
              </li>
            </ul>
          </div>
        </div>
      </>
    );
  }

  renderPandasDataFramePrediction(modelPath: any) {
    return (
      // @ts-expect-error TS(2322): Type '{ position: string; pre: { margin: number; }... Remove this comment to see the full error message
      <div css={styles.item}>
        <Text>
          <FormattedMessage
            defaultMessage="Predict on a Pandas DataFrame:" // eslint-disable-next-line max-len
            description="Section heading to display the code block on how we can use registered model to predict using pandas DataFrame"
          />
        </Text>
        <Paragraph
          dangerouslySetAntdProps={{
            copyable: { text: this.pandasDataFrameCodeText(modelPath) },
          }}
        >
          <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
            <div className="code">
              <span className="code-keyword">import</span> mlflow{`\n`}
              logged_model = <span className="code-string">{`'${modelPath}'`}</span>
            </div>
            <br />
            <div className="code">
              <span className="code-comment">
                {'# '}
                <FormattedMessage
                  defaultMessage="Load model as a PyFuncModel."
                  description="Code comment which states how to load model using PyFuncModel"
                />
              </span>
              {`\n`}
              loaded_model = mlflow.pyfunc.load_model(logged_model)
            </div>
            <br />
            <div className="code">
              <span className="code-comment">
                {'# '}
                <FormattedMessage
                  defaultMessage="Predict on a Pandas DataFrame."
                  // eslint-disable-next-line max-len
                  description="Code comment which states on how we can predict using pandas DataFrame"
                />
              </span>
              {`\n`}
              <span className="code-keyword">import</span> pandas <span className="code-keyword">as</span> pd{`\n`}
              loaded_model.predict(pd.DataFrame(data))
            </div>
          </pre>
        </Paragraph>
      </div>
    );
  }

  renderServingPayload(servingInput?: string) {
    if (servingInput) {
      return (
        <div>
          <span className="code-comment">
            {`# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'\n`}
          </span>
          serving_payload = <span className="code-string">{`"""${servingInput}"""`}</span>
        </div>
      );
    } else {
      return (
        <div className="code">
          <span className="code-comment">
            {`# The logged model does not contain an input_example. 
# Manually generate a serving payload to verify your model prior to deployment.\n`}
          </span>
          <span className="code-keyword">from</span> mlflow.models <span className="code-keyword">import</span>{' '}
          convert_input_example_to_serving_input{`\n\n`}
          <span className="code-comment">
            {`# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction\n`}
          </span>
          serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)
        </div>
      );
    }
  }

  renderValidateServingInput(modelPath: any, servingInput?: string) {
    return (
      // @ts-expect-error TS(2322): Type '{ position: string; pre: { margin: number; }... Remove this comment to see the full error message
      <div css={styles.item}>
        <Text>
          <FormattedMessage
            defaultMessage="Run the following code to validate model inference works on the example payload, prior to deploying it to a serving endpoint" // eslint-disable-next-line max-len
            description="Section heading to display the code block on how we can use validate an input against registered model prior to serving"
          />
        </Text>
        <Paragraph
          dangerouslySetAntdProps={{
            copyable: { text: this.validateModelForServingText(modelPath, servingInput) },
          }}
        >
          <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
            <div className="code">
              <span className="code-keyword">from</span> mlflow.models <span className="code-keyword">import</span>{' '}
              validate_serving_input{`\n\n`}
              model_uri = <span className="code-string">{`'${modelPath}'\n\n`}</span>
              {this.renderServingPayload(servingInput)}
            </div>
            <br />
            <div className="code">
              <span className="code-comment">
                {'# '}
                <FormattedMessage
                  defaultMessage="Validate the serving payload works on the model"
                  description="Code comment which states how to validate serving payload"
                />
              </span>
              {`\n`}
              validate_serving_input(model_uri, serving_payload)
            </div>
            <br />
          </pre>
        </Paragraph>
      </div>
    );
  }

  renderValidateServingInputCodeSnippet() {
    const { runUuid, path } = this.props;
    const modelPath = `runs:/${runUuid}/${path}`;
    return (
      <>
        <Title level={3}>
          <FormattedMessage
            defaultMessage="Validate the model before deployment"
            // eslint-disable-next-line max-len
            description="Heading text for validating the model before deploying it for serving"
          />
        </Title>
        <div className="artifact-logged-model-view-code-content">
          {this.renderValidateServingInput(modelPath, this.state.serving_input)}
        </div>
      </>
    );
  }

  renderPyfuncCodeSnippet() {
    if (this.state.loader_module === 'mlflow.spark') {
      return this.renderMlflowSparkCodeSnippet();
    }
    const { runUuid, path } = this.props;
    const modelPath = `runs:/${runUuid}/${path}`;
    return (
      <>
        <Title level={3}>
          <FormattedMessage
            defaultMessage="Make Predictions"
            // eslint-disable-next-line max-len
            description="Heading text for the prediction section on the registered model from the experiment run"
          />
        </Title>
        <div className="artifact-logged-model-view-code-content">
          {this.renderPandasDataFramePrediction(modelPath)}
          {/* @ts-expect-error TS(2322): Type '{ position: string; pre: { margin: number; }... Remove this comment to see the full error message */}
          <div css={styles.item}>
            <Text>
              <FormattedMessage
                defaultMessage="Predict on a Spark DataFrame:"
                // eslint-disable-next-line max-len
                description="Section heading to display the code block on how we can use registered model to predict using spark DataFrame"
              />
            </Text>
            <Paragraph
              dangerouslySetAntdProps={{
                copyable: { text: this.sparkDataFrameCodeText(modelPath) },
              }}
            >
              <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap', marginTop: 10 }}>
                <div className="code">
                  <span className="code-keyword">import</span> mlflow{`\n`}
                  <span className="code-keyword">from</span> pyspark.sql.functions{' '}
                  <span className="code-keyword">import</span> struct, col{`\n`}
                  logged_model = <span className="code-string">{`'${modelPath}'`}</span>
                </div>
                <br />
                <div className="code">
                  <span className="code-comment">
                    {'# '}
                    <FormattedMessage
                      // eslint-disable-next-line max-len
                      defaultMessage="Load model as a Spark UDF. Override result_type if the model does not return double values."
                      description="Code comment which states how to load model using spark UDF"
                    />
                  </span>
                  {`\n`}
                  loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')
                </div>
                <br />
                <div className="code">
                  <span className="code-comment">
                    {'# '}
                    <FormattedMessage
                      defaultMessage="Predict on a Spark DataFrame."
                      // eslint-disable-next-line max-len
                      description="Code comment which states on how we can predict using spark DataFrame"
                    />
                  </span>
                  {`\n`}
                  df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
                </div>
              </pre>
            </Paragraph>
          </div>
        </div>
      </>
    );
  }

  renderMlflowSparkCodeSnippet() {
    const { runUuid, path } = this.props;
    const modelPath = `runs:/${runUuid}/${path}`;
    return (
      <>
        <Title level={3}>
          <FormattedMessage
            defaultMessage="Make Predictions"
            // eslint-disable-next-line max-len
            description="Heading text for the prediction section on the registered model from the experiment run"
          />
        </Title>
        <div className="artifact-logged-model-view-code-content">
          {this.renderPandasDataFramePrediction(modelPath)}
          {/* @ts-expect-error TS(2322): Type '{ position: string; pre: { margin: number; }... Remove this comment to see the full error message */}
          <div css={styles.item}>
            <Paragraph
              dangerouslySetAntdProps={{
                copyable: { text: this.mlflowSparkCodeText(modelPath) },
              }}
            >
              <pre style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap', marginTop: 10 }}>
                <div className="code">
                  <span className="code-keyword">import</span> mlflow{`\n`}
                  logged_model = <span className="code-string">{`'${modelPath}'`}</span>
                </div>
                <br />
                <div className="code">
                  <span className="code-comment">
                    {'# '}
                    <FormattedMessage
                      // eslint-disable-next-line max-len
                      defaultMessage="Load model"
                      description="Code comment which states how to load a SparkML model"
                    />
                  </span>
                  {`\n`}
                  loaded_model = mlflow.spark.load_model(logged_model)
                </div>
                <br />
                <div className="code">
                  <span className="code-comment">
                    {'# '}
                    <FormattedMessage
                      defaultMessage="Perform inference via model.transform()"
                      // eslint-disable-next-line max-len
                      description="Code comment which states how we can perform SparkML inference"
                    />
                  </span>
                  {`\n`}
                  loaded_model.transform(data)
                </div>
              </pre>
            </Paragraph>
          </div>
        </div>
      </>
    );
  }

  render() {
    if (this.state.loading) {
      return <ArtifactViewSkeleton className="artifact-logged-model-view-loading" />;
    } else if (this.state.error) {
      return (
        <ArtifactViewErrorState
          className="artifact-logged-model-view-error"
          description={
            <FormattedMessage
              defaultMessage="Couldn't load model information due to an error."
              description="Error state text when the model artifact was unable to load"
            />
          }
        />
      );
    } else {
      return (
        <div className="ShowArtifactPage">
          <div className="show-artifact-logged-model-view">
            <div
              className="artifact-logged-model-view-header"
              style={{ marginTop: 16, marginBottom: 16, marginLeft: 16 }}
            >
              <Title level={2}>
                <FormattedMessage defaultMessage="MLflow Model" description="Heading text for mlflow model artifact" />
              </Title>
              {this.state.flavor === 'pyfunc' ? (
                <FormattedMessage
                  // eslint-disable-next-line max-len
                  defaultMessage="The code snippets below demonstrate how to make predictions using the logged model."
                  // eslint-disable-next-line max-len
                  description="Subtext heading explaining the below section of the model artifact view on how users can prediction using the registered logged model"
                />
              ) : (
                <FormattedMessage
                  // eslint-disable-next-line max-len
                  defaultMessage="The code snippets below demonstrate how to load the logged model."
                  // eslint-disable-next-line max-len
                  description="Subtext heading explaining the below section of the model artifact view on how users can load the registered logged model"
                />
              )}{' '}
              {this.renderModelRegistryText()}
            </div>
            <hr />
            <div
              className="artifact-logged-model-view-schema-table"
              style={{ width: '45%', marginLeft: 16, float: 'left' }}
            >
              <Title level={3}>
                <FormattedMessage
                  defaultMessage="Model schema"
                  // eslint-disable-next-line max-len
                  description="Heading text for the model schema of the registered model from the experiment run"
                />
              </Title>
              <div className="content">
                <Text>
                  <FormattedMessage
                    defaultMessage="Input and output schema for your model. <link>Learn more</link>"
                    // eslint-disable-next-line max-len
                    description="Input and output params of the model that is registered from the experiment run"
                    values={{
                      link: (
                        chunks: any, // Reported during ESLint upgrade
                      ) => (
                        // eslint-disable-next-line react/jsx-no-target-blank
                        <a href={ModelSignatureUrl} target="_blank">
                          {chunks}
                        </a>
                      ),
                    }}
                  />
                </Text>
              </div>
              <div style={{ marginTop: 12 }}>
                <SchemaTable schema={{ inputs: this.state.inputs, outputs: this.state.outputs }} defaultExpandAllRows />
              </div>
            </div>
            <div
              className="artifact-logged-model-view-code-group"
              style={{ width: '50%', marginRight: 16, float: 'right' }}
            >
              {this.renderValidateServingInputCodeSnippet()}
              {this.state.flavor === 'pyfunc' ? this.renderPyfuncCodeSnippet() : this.renderNonPyfuncCodeSnippet()}
            </div>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchLoggedModelMetadata() {
    const modelFileLocation = getArtifactLocationUrl(`${this.props.path}/${MLMODEL_FILE_NAME}`, this.props.runUuid);
    this.props
      .getArtifact(modelFileLocation)
      .then((response: any) => {
        const parsedJson = yaml.load(response);
        if (parsedJson.signature) {
          const inputs = Array.isArray(parsedJson.signature.inputs)
            ? parsedJson.signature.inputs
            : JSON.parse(parsedJson.signature.inputs || '[]');

          const outputs = Array.isArray(parsedJson.signature.outputs)
            ? parsedJson.signature.outputs
            : JSON.parse(parsedJson.signature.outputs || '[]');

          this.setState({
            inputs,
            outputs,
          });
        } else {
          this.setState({ inputs: '', outputs: '' });
        }
        if (parsedJson.flavors.mleap) {
          this.setState({ flavor: 'mleap' });
        } else if (parsedJson.flavors.python_function) {
          this.setState({
            flavor: 'pyfunc',
            loader_module: parsedJson.flavors.python_function.loader_module,
          });
        } else {
          this.setState({ flavor: Object.keys(parsedJson.flavors)[0] });
        }
        this.setState({ loading: false });
      })
      .catch((error: any) => {
        this.setState({ error: error, loading: false });
      });
  }

  fetchServingInputExample() {
    const servingInputFileLocation = getArtifactLocationUrl(
      `${this.props.path}/${SERVING_INPUT_FILE_NAME}`,
      this.props.runUuid,
    );
    this.props
      .getArtifact(servingInputFileLocation)
      .then((response: any) => {
        this.setState({ serving_input: response });
      })
      .catch(() => {
        this.setState({ serving_input: null });
      });
  }
}

const styles = {
  item: {
    position: 'relative',
    pre: { margin: 0 },
    '.du-bois-light-typography-copy': {
      position: 'absolute',
      top: 0,
      right: 0,
    },
  },
};

export default ShowArtifactLoggedModelView;
