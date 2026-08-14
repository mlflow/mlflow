import React from 'react';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { isObject } from 'lodash';

interface JsonPreviewProps {
  json: string;
  wrapperStyle?: React.CSSProperties;
  overlayStyle?: React.CSSProperties;
  codeSnippetStyle?: React.CSSProperties;
}

export const JsonPreview: React.FC<React.PropsWithChildren<JsonPreviewProps>> = ({
  json,
  wrapperStyle,
  overlayStyle,
  codeSnippetStyle,
}) => {
  const { formattedJson, isJsonContent } = useFormattedJson(json);

  const defaultWrapperStyle: React.CSSProperties = {
    position: 'relative',
    maxHeight: 'calc(1.5em * 9)',
    overflow: 'hidden',
  };

  const defaultOverlayStyle: React.CSSProperties = {
    position: 'absolute',
    bottom: 0,
    right: 0,
    left: 6,
    height: '2em',
    background: 'linear-gradient(transparent, white)',
  };

  const defaultCodeSnippetStyle: React.CSSProperties = {
    padding: '5px',
    overflowX: 'hidden',
  };

  return (
    <div style={{ ...defaultWrapperStyle, ...wrapperStyle }}>
      {isJsonContent ? (
        <>
          <CodeSnippet language="json" style={{ ...defaultCodeSnippetStyle, ...codeSnippetStyle }}>
            {formattedJson}
          </CodeSnippet>
          <div css={{ ...defaultOverlayStyle, ...overlayStyle }} />
        </>
      ) : (
        <>{json}</>
      )}
    </div>
  );
};

function useFormattedJson(json: string) {
  return React.useMemo(() => {
    try {
      const parsed = JSON.parse(json);
      const isJson = isObject(parsed) && typeof parsed !== 'function' && !(parsed instanceof Date);
      return {
        formattedJson: isJson ? JSON.stringify(parsed, null, 2) : json,
        isJsonContent: isJson,
      };
    } catch (e) {
      return {
        formattedJson: json,
        isJsonContent: false,
      };
    }
  }, [json]);
}

export const FormattedJsonDisplay: React.FC<React.PropsWithChildren<{ json: string }>> = ({ json }) => {
  const { formattedJson, isJsonContent } = useFormattedJson(json);

  return (
    <div css={{ whiteSpace: 'pre-wrap' }}>
      {isJsonContent ? (
        <CodeSnippet language="json" wrapLongLines>
          {formattedJson}
        </CodeSnippet>
      ) : (
        <span>{json}</span>
      )}
    </div>
  );
};
