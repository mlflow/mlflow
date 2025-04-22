import React, { useEffect, useRef } from 'react';

import {
  ModelTraceChildToParentFrameMessage,
  ModelTraceParentToChildFrameMessage,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace, ModelTraceChildToParentFrameMessageType } from '@databricks/web-shared/model-trace-explorer';
import { TableSkeleton, TitleSkeleton } from '@databricks/design-system';
import { Version } from '../../../common/constants';

export const ModelTraceExplorerFrameRenderer = ({
  modelTrace,
  height = 700,
  useLatestVersion = false,
}: {
  modelTrace: ModelTrace;
  height?: number | string;
  useLatestVersion?: boolean;
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [isLoading, setIsLoading] = React.useState(true);

  useEffect(() => {
    const handleMessage = (event: MessageEvent<ModelTraceChildToParentFrameMessageType>) => {
      // only handle messages from the child iframe
      const iframeWindow = iframeRef.current?.contentWindow;
      if (!iframeWindow || event.source !== iframeWindow) {
        return;
      }

      switch (event.data.type) {
        case ModelTraceChildToParentFrameMessage.Ready: {
          setIsLoading(false);
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener('message', handleMessage);
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [modelTrace]);

  useEffect(() => {
    const iframeWindow = iframeRef.current?.contentWindow;
    if (!iframeWindow || isLoading) {
      return;
    }

    iframeWindow.postMessage({
      type: ModelTraceParentToChildFrameMessage.UpdateTrace,
      traceData: modelTrace,
    });
  }, [modelTrace, isLoading]);

  return (
    <div css={{ height }}>
      {isLoading && (
        <div
          css={{
            position: 'absolute',
            width: '100%',
            height,
          }}
        >
          <TitleSkeleton />
          <TableSkeleton lines={5} />
        </div>
      )}
      <iframe
        title="Model Trace Explorer"
        // Include the current mlflow version as a query parameter to bust the browser cache
        // generated and prevent https://github.com/mlflow/mlflow/issues/13829.
        src={`static-files/lib/ml-model-trace-renderer/index.html?version=${Version}`}
        ref={iframeRef}
        css={{
          border: 'none',
          width: '100%',
          height,
        }}
      />
    </div>
  );
};
