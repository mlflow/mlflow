import { useMemo } from 'react';

export const EvaluationsReviewExpandedJSONValueCell = ({ value }: { value: string | Record<string, unknown> }) => {
  const structuredJSONValue = useMemo(() => {
    // If value is already an object, stringify it directly
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2);
    }

    // If value is a string, try to parse it as JSON
    if (typeof value === 'string') {
      try {
        const objectData = JSON.parse(value);
        return JSON.stringify(objectData, null, 2);
      } catch (e) {
        return null;
      }
    }

    // For any other type, return null
    return null;
  }, [value]);

  return (
    <div
      css={{
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontFamily: structuredJSONValue ? 'monospace' : undefined,
      }}
    >
      {structuredJSONValue || String(value)}
    </div>
  );
};
