import { ErrorView } from './ErrorView';

export const PageNotFoundView = () => {
  return <ErrorView statusCode={404} fallbackHomePageReactRoute="/" />;
};

export default PageNotFoundView;
