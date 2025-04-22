import { createRoot } from 'react-dom/client';
import { MLFlowRoot } from './app';

// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
const root = createRoot(document.getElementById('root')!);
root.render(<MLFlowRoot />);

const windowOnError = (message: Event | string, source?: string, lineno?: number, colno?: number, error?: Error) => {
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.error(error, message);
  // returning false allows the default handler to fire as well
  return false;
};

window.onerror = windowOnError;
