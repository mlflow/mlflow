import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { UpdateExperimentViewStateFn } from '../../../../types';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';

/**
 * Component used to expand/collapse runs list (table) when in compare runs mode.
 */
export const ExperimentViewRunsTableCollapse = ({
  updateViewState,
  runListHidden,
}: {
  updateViewState: UpdateExperimentViewStateFn;
  runListHidden: SearchExperimentRunsViewState['runListHidden'];
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        width: 2 * theme.spacing.md,
        right: -theme.spacing.md,
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'transparent',
        '&:hover': {
          opacity: 1,
          '.bar': { opacity: 1 },
          '.button': {
            border: `2px solid ${theme.colors.primary}`,
          },
        },
        opacity: runListHidden ? 1 : 0,
        transition: 'opacity 0.2s',
      }}
    >
      <div
        className='bar'
        css={{
          transition: 'opacity 0.2s',
          opacity: 0,
          position: 'absolute',
          left: theme.spacing.md - 2,
          top: 0,
          backgroundColor: theme.colors.primary,
          bottom: 0,
          width: 3,
          pointerEvents: 'none',
        }}
      />
      <div
        className='button'
        css={{
          transition: 'border-color 0.2s',
          position: 'relative',
          width: theme.general.iconSize,
          height: theme.general.iconSize,
          backgroundColor: theme.colors.backgroundPrimary,
          borderRadius: theme.general.iconSize,
          overflow: 'hidden',
          border: `1px solid ${theme.colors.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Button
          onClick={() => updateViewState({ runListHidden: !runListHidden })}
          icon={runListHidden ? <ChevronRightIcon /> : <ChevronLeftIcon />}
          size='small'
        />
      </div>
    </div>
  );
};
