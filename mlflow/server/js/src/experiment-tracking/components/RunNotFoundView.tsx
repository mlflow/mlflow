import Routes from '../routes';
import { ErrorView } from '../../common/components/ErrorView';

type Props = {
  runId: string;
};

export function RunNotFoundView({ runId }: Props) {
  return (
    <ErrorView
      statusCode={404}
      subMessage={`Run ID ${runId} does not exist`}
      fallbackHomePageReactRoute={Routes.rootRoute}
    />
  );
}
