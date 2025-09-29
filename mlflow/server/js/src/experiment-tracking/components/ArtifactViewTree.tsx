// @ts-expect-error TS(7016): Could not find a declaration file for module 'reac... Remove this comment to see the full error message
import type { TreebeardData } from 'react-treebeard';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'reac... Remove this comment to see the full error message
import { decorators, Treebeard } from 'react-treebeard';
import { DATA_EXTENSIONS, getExtension, IMAGE_EXTENSIONS, TEXT_EXTENSIONS } from '../../common/utils/FileUtils';

import spinner from '../../common/static/mlflow-spinner.png';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import type { Theme } from '@emotion/react';

interface ArtifactViewTreeProps {
  onToggleTreebeard: (
    dataNode: {
      id: string;
      loading: boolean;
    },
    toggled: boolean,
  ) => void;
  data: TreebeardData;
}

export const ArtifactViewTree = ({ data, onToggleTreebeard }: ArtifactViewTreeProps) => {
  const { theme } = useDesignSystemTheme();
  const treebeardStyle = useMemo(() => getTreebeardStyle(theme), [theme]);
  return <Treebeard data={data} onToggle={onToggleTreebeard} style={treebeardStyle} decorators={decorators} />;
};

interface DecoratorStyle {
  style: React.CSSProperties & {
    base: React.CSSProperties;
    title: React.CSSProperties;
  };
  node: {
    name: string;
    children: string[];
  };
}
decorators.Header = ({ style, node }: DecoratorStyle) => {
  let iconType;
  if (node.children) {
    iconType = 'folder';
  } else {
    const extension = getExtension(node.name);
    if (IMAGE_EXTENSIONS.has(extension)) {
      iconType = 'file-image-o';
    } else if (DATA_EXTENSIONS.has(extension)) {
      iconType = 'file-excel-o';
    } else if (TEXT_EXTENSIONS.has(extension)) {
      iconType = 'file-code-o';
    } else {
      iconType = 'file-text-o';
    }
  }
  const iconClass = `fa fa-${iconType}`;

  // Add margin-left to the non-directory nodes to align the arrow, icons, and texts.
  const iconStyle = node.children ? { marginRight: '5px' } : { marginRight: '5px', marginLeft: '19px' };

  return (
    <div
      style={style.base}
      data-testid="artifact-tree-node"
      // eslint-disable-next-line react/no-unknown-property
      artifact-name={node.name}
      aria-label={node.name}
    >
      <div style={style.title}>
        <i className={iconClass} style={iconStyle} />
        {node.name}
      </div>
    </div>
  );
};

decorators.Loading = ({ style }: DecoratorStyle) => {
  return (
    <div style={style}>
      <img alt="" className="mlflow-loading-spinner" src={spinner} />
      <FormattedMessage
        defaultMessage="loading..."
        description="Loading spinner text to show that the artifact loading is in progress"
      />
    </div>
  );
};

const getTreebeardStyle = (theme: Theme) => ({
  tree: {
    base: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      backgroundColor: theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      fontSize: theme.typography.fontSizeMd,
      maxWidth: '500px',
      height: '100%',
      overflow: 'scroll',
    },
    node: {
      base: {
        position: 'relative',
      },
      link: {
        cursor: 'pointer',
        position: 'relative',
        padding: '0px 5px',
        display: 'block',
      },
      activeLink: {
        background: theme.isDarkMode ? theme.colors.grey700 : theme.colors.grey300,
      },
      toggle: {
        base: {
          position: 'relative',
          display: 'inline-block',
          verticalAlign: 'top',
          marginLeft: '-5px',
          height: '24px',
          width: '24px',
        },
        wrapper: {
          position: 'absolute',
          top: '50%',
          left: '50%',
          margin: '-12px 0 0 -4px',
          height: '14px',
          display: 'flex',
          alignItems: 'end',
        },
        height: 7,
        width: 7,
        arrow: {
          fill: '#7a7a7a',
          strokeWidth: 0,
        },
      },
      header: {
        base: {
          display: 'inline-block',
          verticalAlign: 'top',
          color: theme.colors.textPrimary,
        },
        connector: {
          width: '2px',
          height: '12px',
          borderLeft: 'solid 2px black',
          borderBottom: 'solid 2px black',
          position: 'absolute',
          top: '0px',
          left: '-21px',
        },
        title: {
          lineHeight: '24px',
          verticalAlign: 'middle',
        },
      },
      subtree: {
        listStyle: 'none',
        paddingLeft: '19px',
      },
    },
  },
});
