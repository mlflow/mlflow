import React from 'react';
import Admonition from '@theme/Admonition';

const DATABRICKS_DOCS_BASE_URL = 'https://docs.databricks.com/aws/en/mlflow';
const DATABRICKS_SIGNUP_URL = 'https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=TRY_MLFLOW&dbx_source=TRY_MLFLOW&signup_experience_step=EXPRESS&provider=MLFLOW';

interface FeedbackComingSoonProps {
  docsPath: string;
  additionalText?: string;
}

const FeedbackComingSoon: React.FC<FeedbackComingSoonProps> = ({
  docsPath,
  additionalText
}) => {
  const fullDocsUrl: string = `${DATABRICKS_DOCS_BASE_URL}${docsPath}`;

  return (
    <Admonition type="warning" title="Features Coming Soon to OSS MLflow">
      <p>The features discussed on this page are currently only available in Managed MLflow on Databricks, but are coming soon to OSS MLflow.</p>
      
      <p>
        For immediate access to these features, visit this page{' '}
        <a href={fullDocsUrl} target="_blank" rel="noopener noreferrer">
          in the Databricks MLflow documentation
        </a>{' '}
        to learn more about using these features in Databricks.
      </p>
      
      {additionalText && <p>{additionalText}</p>}
      
      <p>
        Want to try out the full capabilities today? You can{' '}
        <a href={DATABRICKS_SIGNUP_URL} target="_blank" rel="noopener noreferrer">
          try Databricks for free here
        </a>.
      </p>
    </Admonition>
  );
};

export default FeedbackComingSoon;