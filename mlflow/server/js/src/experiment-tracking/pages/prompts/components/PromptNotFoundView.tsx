import { ErrorView } from '@mlflow/mlflow/src/common/components/ErrorView';
import Routes from '../../../routes';

interface Props {
  promptName: string;
}

export function PromptNotFoundView({ promptName }: Props) {
  return (
    <ErrorView
      statusCode={404}
      subMessage={`Prompt name '${promptName}' does not exist`}
      fallbackHomePageReactRoute={Routes.promptsPageRoute}
    />
  );
}
