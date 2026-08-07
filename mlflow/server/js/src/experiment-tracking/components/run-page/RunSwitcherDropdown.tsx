import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChartIcon,
  Button,
  ChevronDownIcon,
  CloseIcon,
  DropdownMenu,
  Input,
  SearchIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { MlflowService } from '../../sdk/MlflowService';
import type { RunEntity } from '../../types';

const MAX_VISIBLE_RUNS = 25;

const RunSwitcherBody = ({
  runs,
  loading,
  currentRunUuid,
  comparisonRunUuids,
  onSelect,
  onCompareRun,
}: {
  runs: RunEntity[];
  loading: boolean;
  currentRunUuid: string;
  comparisonRunUuids?: string[];
  onSelect: (runUuid: string) => void;
  onCompareRun?: (run: RunEntity) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const inputRef = useRef<React.ComponentRef<typeof Input>>(null);
  const firstItemRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState('');

  const filteredRuns = useMemo(
    () => runs.filter((r) => r.info.runName?.toLowerCase().includes(filter.toLowerCase())).slice(0, MAX_VISIBLE_RUNS),
    [runs, filter],
  );

  useEffect(() => {
    requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
  }, []);

  return (
    <>
      <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.sm}px`, width: '100%' }}>
        <Input
          componentId="mlflow.run-page.run-switcher.search"
          prefix={<SearchIcon />}
          value={filter}
          type="search"
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search runs"
          autoFocus
          ref={inputRef}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' || e.key === 'Tab') {
              firstItemRef.current?.focus();
              return;
            }
            e.stopPropagation();
          }}
        />
      </div>
      <DropdownMenu.Group css={{ maxHeight: 300, overflowY: 'auto' }}>
        {loading && (
          <DropdownMenu.Item componentId="mlflow.run-page.run-switcher.loading" disabled>
            Loading…
          </DropdownMenu.Item>
        )}
        {!loading &&
          filteredRuns.map((run, index) => {
            const isCurrentRun = run.info.runUuid === currentRunUuid;
            const isComparisonRun = comparisonRunUuids?.includes(run.info.runUuid) ?? false;
            const atMax = (comparisonRunUuids?.length ?? 0) >= 5;
            return (
              <DropdownMenu.CheckboxItem
                componentId="mlflow.run-page.run-switcher.item"
                key={run.info.runUuid}
                checked={isCurrentRun}
                onClick={() => onSelect(run.info.runUuid)}
                ref={index === 0 ? firstItemRef : undefined}
              >
                <DropdownMenu.ItemIndicator />
                <span css={{ flex: 1 }}>{run.info.runName || run.info.runUuid}</span>
                {onCompareRun && !isCurrentRun && (
                  <Tooltip
                    componentId="mlflow.run-page.run-switcher.compare-tooltip"
                    side="right"
                    content={
                      isComparisonRun
                        ? 'Remove comparison'
                        : atMax
                          ? 'Maximum 5 comparisons reached'
                          : 'Overlay metrics'
                    }
                  >
                    <Button
                      componentId="mlflow.run-page.run-switcher.compare-btn"
                      size="small"
                      type={isComparisonRun ? 'primary' : 'tertiary'}
                      disabled={!isComparisonRun && atMax}
                      icon={isComparisonRun ? <CloseIcon /> : <BarChartIcon />}
                      onClick={(e) => {
                        e.stopPropagation();
                        onCompareRun(run);
                      }}
                    />
                  </Tooltip>
                )}
              </DropdownMenu.CheckboxItem>
            );
          })}
        {!loading && filteredRuns.length === 0 && (
          <DropdownMenu.Item componentId="mlflow.run-page.run-switcher.no-results" disabled>
            No runs found
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Group>
    </>
  );
};

export const RunSwitcherDropdown = ({
  experimentId,
  currentRunUuid,
  activeTab,
  comparisonRunUuids,
  onCompareRun,
  onClearComparisons,
}: {
  experimentId: string;
  currentRunUuid: string;
  activeTab: string;
  comparisonRunUuids?: string[];
  onCompareRun?: (run: RunEntity) => void;
  onClearComparisons?: () => void;
}) => {
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);
  const [runs, setRuns] = useState<RunEntity[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasFetched, setHasFetched] = useState(false);

  const comparisonLabel = useMemo(() => {
    if (!comparisonRunUuids?.length) return null;
    if (comparisonRunUuids.length === 1) {
      return runs.find((r) => r.info.runUuid === comparisonRunUuids[0])?.info.runName ?? comparisonRunUuids[0];
    }
    return `${comparisonRunUuids.length} runs`;
  }, [runs, comparisonRunUuids]);

  const handleOpenChange = (nextOpen: boolean) => {
    setOpen(nextOpen);
    if (nextOpen && !hasFetched) {
      setLoading(true);
      MlflowService.searchRuns({ experiment_ids: [experimentId], max_results: 200 })
        .then((response) => {
          setRuns(response.runs ?? []);
        })
        .catch(() => {
          setRuns([]);
        })
        .finally(() => {
          setLoading(false);
          setHasFetched(true);
        });
    }
  };

  const handleSelect = (runUuid: string) => {
    setOpen(false);
    navigate(Routes.getRunPageTabRoute(experimentId, runUuid, activeTab));
  };

  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
      <DropdownMenu.Root open={open} onOpenChange={handleOpenChange} modal={false}>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.run-page.run-switcher.trigger"
            size="small"
            icon={<ChevronDownIcon />}
            aria-label="Switch run"
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content minWidth={260}>
          <RunSwitcherBody
            runs={runs}
            loading={loading}
            currentRunUuid={currentRunUuid}
            comparisonRunUuids={comparisonRunUuids}
            onSelect={handleSelect}
            onCompareRun={onCompareRun}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      {comparisonLabel && onClearComparisons && (
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary" size="sm">
            vs
          </Typography.Text>
          <Typography.Text size="sm" bold>
            {comparisonLabel}
          </Typography.Text>
          <Button
            componentId="mlflow.run-page.run-switcher.clear-compare"
            size="small"
            type="tertiary"
            icon={<CloseIcon />}
            onClick={onClearComparisons}
            aria-label="Clear comparison"
          />
        </span>
      )}
    </span>
  );
};
