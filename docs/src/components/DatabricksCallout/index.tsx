import React from 'react';
import Admonition from '@theme/Admonition';

const DATABRICKS_DOCS_BASE_URL = 'https://docs.databricks.com/aws/en/mlflow';
export const DATABRICKS_SIGNUP_URL = 'https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=TRY_MLFLOW&dbx_source=TRY_MLFLOW&signup_experience_step=EXPRESS';

interface DatabricksCalloutProps {
  docsPath: string;
  additionalText?: string;
}


const DatabricksCallout: React.FC<DatabricksCalloutProps> = ({
  docsPath,
  additionalText
}) => {
  const fullDocsUrl: string = `${DATABRICKS_DOCS_BASE_URL}${docsPath}`;

  return (
    <Admonition type="tip" title="Building GenAI Apps in Databricks">
      <p>Are you looking for guidance on building your GenAI apps in Databricks?</p>
      
      <p>
        Visit this page{' '}
        <a href={fullDocsUrl} target="_blank" rel="noopener noreferrer">
          in the managed MLflow documentation
        </a>{' '}
        to learn more!
      </p>
      
      {additionalText && <p>{additionalText}</p>}
      
      <p>
        Are you interested in trying out a fully-featured managed MLflow solution for GenAI? You can{' '}
        <a href={DATABRICKS_SIGNUP_URL} target="_blank" rel="noopener noreferrer">
          try Databricks for free here
        </a>.
      </p>
    </Admonition>
  );
};

export default DatabricksCallout;
