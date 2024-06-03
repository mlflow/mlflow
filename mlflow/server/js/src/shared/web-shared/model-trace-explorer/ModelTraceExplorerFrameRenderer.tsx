import React, { useEffect, useRef } from 'react';

import {
  ModelTraceChildToParentFrameMessage,
  ModelTraceParentToChildFrameMessage,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace, ModelTraceChildToParentFrameMessageType } from '@databricks/web-shared/model-trace-explorer';
import { TableSkeleton, TitleSkeleton } from '@databricks/design-system';

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
        src="static-files/lib/ml-model-trace-renderer/index.html"
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
