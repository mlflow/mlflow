import React, { Component, useCallback, useEffect, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter, createElement } from 'react-syntax-highlighter';
import { coy as style, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { useVirtualizer } from '@tanstack/react-virtual';
import { getExtension, getLanguage } from '../../../common/utils/FileUtils';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';
import type { DesignSystemHocProps } from '@databricks/design-system';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

const LARGE_ARTIFACT_SIZE = 100 * 1024;
// react-syntax-highlighter's default renderer flattens its per-line tree with a single
// `[].concat(...lines)` call, which overflows V8's ~125K argument limit and throws
// `RangeError: Maximum call stack size exceeded` for line-heavy files. Above this line
// count we switch to a virtualized renderer, which both avoids that call and keeps the
// DOM to ~the visible lines instead of one element per line.
const VIRTUALIZATION_LINE_THRESHOLD = 5000;
const ESTIMATED_LINE_HEIGHT = 20;
const VIRTUALIZED_LINE_OVERSCAN = 25;

type SyntaxHighlighterRendererProps = {
  rows: any[];
  stylesheet: any;
  useInlineStyles: boolean;
};

type VirtualizedRowsProps = SyntaxHighlighterRendererProps & {
  scrollElementRef: React.RefObject<HTMLPreElement>;
};

// Renders only the visible window of highlighted rows. The virtualizer lives here rather
// than in VirtualizedSyntaxHighlighter so that scrolling re-renders only this component,
// not <SyntaxHighlighter />, whose render re-processes the entire file.
const VirtualizedRows = ({ rows, stylesheet, useInlineStyles, scrollElementRef }: VirtualizedRowsProps) => {
  // Resolve the scroll element in an effect: this component mounts before the ref to its
  // ancestor <pre> is attached, so reading scrollElementRef.current during render (or in
  // getScrollElement directly) would leave the virtualizer permanently unmeasured.
  const [scrollElement, setScrollElement] = useState<HTMLPreElement | null>(null);
  useEffect(() => {
    setScrollElement(scrollElementRef.current);
  }, [scrollElementRef]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => ESTIMATED_LINE_HEIGHT,
    overscan: VIRTUALIZED_LINE_OVERSCAN,
  });

  return (
    <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
      {virtualizer.getVirtualItems().map((virtualLine) => (
        <div
          key={virtualLine.key}
          data-index={virtualLine.index}
          ref={virtualizer.measureElement}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            transform: `translateY(${virtualLine.start}px)`,
          }}
        >
          {createElement({
            node: rows[virtualLine.index],
            stylesheet,
            useInlineStyles,
            key: `line-${virtualLine.index}`,
          })}
        </div>
      ))}
    </div>
  );
};

type VirtualizedSyntaxHighlighterProps = {
  language: string;
  style: any;
  customStyle: React.CSSProperties;
  children: string;
};

// react-syntax-highlighter forwards unrecognized props to PreTag, which lets us attach
// a ref to the scroll container without defining a component during render.
const PreWithScrollRef = ({
  scrollRef,
  ...props
}: React.HTMLProps<HTMLPreElement> & { scrollRef: React.RefObject<HTMLPreElement> }) => (
  <pre {...props} ref={scrollRef} />
);

// Drop-in replacement for <SyntaxHighlighter /> for line-heavy files. Providing a custom
// renderer makes react-syntax-highlighter keep per-line rows (wrapLines) instead of
// flattening them through the argument-limited `[].concat(...)` call, and the renderer
// mounts only the visible lines.
const VirtualizedSyntaxHighlighter = ({
  language,
  style,
  customStyle,
  children,
}: VirtualizedSyntaxHighlighterProps) => {
  const preRef = useRef<HTMLPreElement>(null);

  // The renderer identity must be stable across renders, otherwise the rendered rows
  // are remounted on every render and the scroll container loses its scroll position.
  const renderer = useCallback(
    ({ rows, stylesheet, useInlineStyles }: SyntaxHighlighterRendererProps) => (
      <VirtualizedRows
        rows={rows}
        stylesheet={stylesheet}
        useInlineStyles={useInlineStyles}
        scrollElementRef={preRef}
      />
    ),
    [],
  );

  return (
    <SyntaxHighlighter
      language={language}
      style={style}
      customStyle={customStyle}
      PreTag={PreWithScrollRef}
      scrollRef={preRef}
      renderer={renderer}
    >
      {children}
    </SyntaxHighlighter>
  );
};

// Exported for tests
export function isLineHeavy(text: string): boolean {
  let lines = 1;
  for (let i = text.indexOf('\n'); i !== -1 && lines <= VIRTUALIZATION_LINE_THRESHOLD; i = text.indexOf('\n', i + 1)) {
    lines++;
  }
  return lines > VIRTUALIZATION_LINE_THRESHOLD;
}

type Props = DesignSystemHocProps & {
  runUuid: string;
  path: string;
  size?: number;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

type State = {
  loading?: boolean;
  error?: Error;
  text?: string;
  path?: string;
};

class ShowArtifactTextView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static defaultProps = {
    getArtifact: fetchArtifactUnified,
  };

  state = {
    loading: true,
    error: undefined,
    text: undefined,
    path: undefined,
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps: Props) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading || this.state.path !== this.props.path) {
      return <ArtifactViewSkeleton className="artifact-text-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState className="artifact-text-view-error" />;
    } else {
      const isLargeFile = (this.props.size || 0) > LARGE_ARTIFACT_SIZE;
      const language = isLargeFile ? 'text' : getLanguage(this.props.path);
      const { theme } = this.props.designSystemThemeApi;

      const overrideStyles = {
        fontFamily: 'Source Code Pro,Menlo,monospace',
        fontSize: theme.typography.fontSizeMd,
        overflow: 'auto',
        marginTop: '0',
        width: '100%',
        height: '100%',
        padding: theme.spacing.xs,
        borderColor: theme.colors.borderDecorative,
        border: 'none',
      };
      const renderedContent = this.state.text
        ? prettifyArtifactText(language, this.state.text, this.props.path)
        : this.state.text;

      const syntaxStyle = theme.isDarkMode ? darkStyle : style;
      const text = renderedContent ?? '';
      const TextSyntaxHighlighter = isLineHeavy(text) ? VirtualizedSyntaxHighlighter : SyntaxHighlighter;

      return (
        <div className="mlflow-ShowArtifactPage">
          <div className="text-area-border-box">
            <TextSyntaxHighlighter language={language} style={syntaxStyle} customStyle={overrideStyles}>
              {text}
            </TextSyntaxHighlighter>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    this.setState({ loading: true });
    const { isLoggedModelsMode, loggedModelId, path, runUuid, experimentId, entityTags } = this.props;

    this.props
      .getArtifact?.({ isLoggedModelsMode, loggedModelId, path, runUuid, experimentId, entityTags }, getArtifactContent)
      .then((text: string) => {
        this.setState({ text: text, loading: false });
      })
      .catch((error: Error) => {
        this.setState({ error: error, loading: false });
      });
    this.setState({ path: this.props.path });
  }
}

export function prettifyArtifactText(language: string, rawText: string, path?: string) {
  if (path && getExtension(path).toLowerCase() === 'jsonl') {
    return rawText;
  }
  if (language === 'json') {
    try {
      const parsedJson = JSON.parse(rawText);
      return JSON.stringify(parsedJson, null, 2);
    } catch (e) {
      return rawText;
    }
  }
  return rawText;
}
export default React.memo(WithDesignSystemThemeHoc(ShowArtifactTextView));
