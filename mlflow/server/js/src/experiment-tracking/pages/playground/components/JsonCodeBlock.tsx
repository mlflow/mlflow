import { Button, ChevronDownIcon, ChevronUpIcon, CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { useEffect, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '../../../../shared/building_blocks/CopyButton';
import { prettyPrintJson } from '../utils';

// Collapse the code block once it grows past this height and reveal the rest
// behind a "See more" toggle, so a large JSON payload does not dominate the card.
const COLLAPSED_MAX_HEIGHT = 640;

interface Props {
  /** Raw JSON string (e.g. tool-call arguments or a JSON-format reply). */
  raw: string;
}

/**
 * Renders a JSON string as a syntax-highlighted code block with line numbers and
 * a copy button. The JSON is pretty-printed when parseable and shown verbatim
 * otherwise. The snippet background is transparent so the surrounding assistant
 * card background shows through (the shared `CodeSnippet` prism theme would
 * otherwise paint its own opaque background). When the rendered output is taller
 * than {@link COLLAPSED_MAX_HEIGHT}, it is collapsed behind a "See more" toggle.
 */
export const JsonCodeBlock = ({ raw }: Props) => {
  const { theme } = useDesignSystemTheme();
  const code = prettyPrintJson(raw);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const el = viewportRef.current;
    if (el) {
      // scrollHeight is the full content height regardless of the collapsed clamp,
      // so this stays correct whether the block is expanded or not.
      setIsOverflowing(el.scrollHeight > COLLAPSED_MAX_HEIGHT);
    }
  }, [code]);

  const collapsed = isOverflowing && !expanded;

  return (
    <div css={{ position: 'relative' }} data-testid="mlflow.playground.json_code_block">
      <div css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}>
        <CopyButton
          componentId="mlflow.playground.json_code_block.copy"
          copyText={code}
          showLabel={false}
          size="small"
          icon={<CopyIcon />}
        />
      </div>
      <div ref={viewportRef} css={{ maxHeight: collapsed ? COLLAPSED_MAX_HEIGHT : 'none', overflow: 'hidden' }}>
        <CodeSnippet
          theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
          language="json"
          showLineNumbers
          style={{
            margin: 0,
            padding: 0,
            // Transparent so the assistant card background shows through instead of
            // the prism theme's own (dark "purple") background.
            backgroundColor: 'transparent',
          }}
        >
          {code}
        </CodeSnippet>
      </div>
      {isOverflowing && (
        <Button
          componentId="mlflow.playground.json_code_block.toggle"
          css={{ width: '100%' }}
          type="tertiary"
          size="small"
          icon={expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? (
            <FormattedMessage
              defaultMessage="See less"
              description="Toggle that collapses a long JSON code block on the playground page"
            />
          ) : (
            <FormattedMessage
              defaultMessage="See more"
              description="Toggle that expands a collapsed long JSON code block on the playground page"
            />
          )}
        </Button>
      )}
    </div>
  );
};
