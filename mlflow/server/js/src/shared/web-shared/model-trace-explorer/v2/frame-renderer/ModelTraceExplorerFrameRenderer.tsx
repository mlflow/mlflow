import React, { useEffect, useRef } from 'react';

import { useDesignSystemTheme, Typography, ChevronDownIcon, ChevronRightIcon, Button } from '@databricks/design-system';
import type { ModelTrace } from '../ModelTrace.types';
import type { ModelTraceChildToParentFrameMessageType } from '../../frame-renderer/types';
import { ModelTraceExplorerSkeleton } from '../../ModelTraceExplorerSkeleton';
import { ModelTraceChildToParentFrameMessage, ModelTraceParentToChildFrameMessage } from '../../frame-renderer/types';

import { findTraceVersionByKey, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import rendererVersions from '../../ml-model-trace-renderer/library-versions.json';

const FALLBACK_VERSION = '2';

const TITLE_BAR_HEIGHT = 36;

export const getTraceVersion = (modelTrace: ModelTrace): string => {
  // retrieve version from v3 info
  if (isV3ModelTraceInfo(modelTrace?.info)) {
    return findTraceVersionByKey(modelTrace?.info?.trace_metadata) ?? FALLBACK_VERSION;
  }

  // attempt to retrieve the version from request metadata
  const versionFromRequest = findTraceVersionByKey(modelTrace?.info?.request_metadata);
  if (versionFromRequest) {
    return versionFromRequest;
  }

  // if that fails, attempt to retrieve the version from tags
  // (this is the old way of storing the version). if it's not
  // here either, then this is a pre-public preview trace that
  // should be renderable with the fallback.
  return findTraceVersionByKey(modelTrace?.info?.tags) ?? FALLBACK_VERSION;
};

export const ModelTraceExplorerFrameRenderer = ({
  modelTrace,
  height = 700,
  useLatestVersion = false,
  sqlWarehouseId,
}: {
  modelTrace: ModelTrace;
  height?: number | string;
  useLatestVersion?: boolean;
  sqlWarehouseId?: string;
}): JSX.Element => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [expanded, setExpanded] = React.useState(true);
  const version = useLatestVersion ? 'current' : getTraceVersion(modelTrace);
  const rendererVersionSrc = (rendererVersions as any)[version]?.path ?? rendererVersions[FALLBACK_VERSION].path;
  const fullIframeSrc = sqlWarehouseId ? `${rendererVersionSrc}?sqlWarehouseId=${sqlWarehouseId}` : rendererVersionSrc;

  const { theme } = useDesignSystemTheme();

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
          iframeWindow.postMessage({
            type: ModelTraceParentToChildFrameMessage.UpdateTrace,
            traceData: modelTrace,
          });
          break;
        }
        case ModelTraceChildToParentFrameMessage.LogError: {
          // Surface errors reported by the renderer iframe. Databricks routes these to its panel
          // boundary telemetry; OSS has no such sink, so log to the console for debuggability.
          const { error } = event.data || {};
          // eslint-disable-next-line no-console
          console.error('MLflow trace renderer error:', error);

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
    <div
      css={{
        height: expanded ? height : TITLE_BAR_HEIGHT,
        borderTop: `1px solid ${theme.colors.border}`,
      }}
    >
      <div
        css={{
          display: 'flex',
          height: TITLE_BAR_HEIGHT,
          justifyContent: 'space-between',
          padding: theme.spacing.sm,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Button
            componentId={`mlflow.notebook.trace-ui-${expanded ? 'collapse' : 'expand'}`}
            icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            size="small"
            onClick={() => setExpanded(!expanded)}
          />
          <Typography.Title level={4} withoutMargins>
            MLflow Trace UI
          </Typography.Title>
          <Typography.Link
            componentId="mlflow.notebook.trace-ui-learn-more-link"
            href="https://mlflow.org/docs/latest/llms/tracing/index.html"
            openInNewTab
            title="Learn More"
          >
            Learn More
          </Typography.Link>
        </div>
      </div>
      {isLoading && (
        <div
          css={{
            position: 'absolute',
            width: '100%',
            height,
          }}
        >
          <ModelTraceExplorerSkeleton label="Frame Skeleton" />
        </div>
      )}
      <iframe
        title="Model Trace Explorer"
        src={fullIframeSrc}
        ref={iframeRef}
        css={{
          border: 'none',
          width: '100%',
          height: `calc(100% - ${TITLE_BAR_HEIGHT}px)`,
          display: expanded ? 'block' : 'none',
        }}
      />
    </div>
  );
};
