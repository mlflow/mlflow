import useBaseUrl from '@docusaurus/useBaseUrl';
import { useLocation } from '@docusaurus/router';

const DATABRICKS_DOCS_BASE = 'https://docs.databricks.com/aws/en/mlflow';
const DATABRICKS_DOCS_GENAI = 'https://docs.databricks.com/aws/en/mlflow3/genai/';

interface DatabricksDocsLinkProps {
  mobile?: boolean;
  [key: string]: any;
}

export default function DatabricksDocsLink({ mobile, ...props }: DatabricksDocsLinkProps): JSX.Element {
  const location = useLocation();
  const genaiPath = useBaseUrl('/genai');

  const href = location.pathname.startsWith(genaiPath) ? DATABRICKS_DOCS_GENAI : DATABRICKS_DOCS_BASE;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="navbar__item navbar__link databricks-docs-link"
      {...props}
    >
      Databricks
    </a>
  );
}
