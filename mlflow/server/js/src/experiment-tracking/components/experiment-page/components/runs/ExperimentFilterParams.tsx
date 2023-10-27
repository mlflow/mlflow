import React, { useState, useCallback, useEffect } from 'react';
import { Theme } from '@emotion/react';
import { Input } from '@databricks/design-system';
import { UpdateExperimentSearchFacetsFn } from '../../../../types';
import { makeCanonicalSortKey } from '../../utils/experimentPage.column-utils';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { COLUMN_TYPES } from '../../../../constants';

const FILTER_PARAMS_PLACEHOLDER = 'alpha, lr';

export type ExperimentFilterParamsProps = {
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  searchFacetsState: SearchExperimentRunsFacetsState;
};

const ExperimentFilterParams = (props: ExperimentFilterParamsProps) => {
  const { updateSearchFacets, searchFacetsState } = props;

  const [text, setText] = useState<string>('');
  const [placeholder, setPlaceholder] = useState<string>('');

  const regex = /params\.`(.*?)`/;

  useEffect(() => {
    const { filterParams } = searchFacetsState;

    if (filterParams !== null && filterParams.length > 0) {
      // setText(filterParams.map((item) => regex.exec(item)[1]).join(','));

      const result = filterParams.map((item) => {
        const match = regex.exec(item);
        if (match && match[1]) {
          return match[1];
        }
        return '';
      });

      const joinedResult = result.join(',');

      setText(joinedResult);
    }

    if (text.length === 0){
      setPlaceholder(FILTER_PARAMS_PLACEHOLDER);
    }
    // Add other conditions for updating the placeholder as needed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchFacetsState]);

  const triggerFilter: React.KeyboardEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      if (e.key === 'Enter') {
        if (text.length > 0) {
          const params = text.split(',').map((param) => {
            const trimmedParam = param.trim();
            return makeCanonicalSortKey(COLUMN_TYPES.PARAMS, trimmedParam);
          });

          updateSearchFacets({ filterParams: params });
        } else {
          updateSearchFacets({ filterParams: [] });
        }
      }
    },
    [text, updateSearchFacets],
  );

  return (
    <div css={styles.filterWrapper}>
      <label css={styles.filterLabel}>Filter Params:</label>
      <div>
        <Input
          type='text'
          css={{ width: 560 }}
          placeholder={placeholder}
          value={text}
          onKeyDown={triggerFilter}
          onChange={(e) => setText(e.target.value)}
        />
      </div>
    </div>
  );
};

const styles = {
  filterWrapper: (theme: Theme) => ({ display: 'flex', gap: theme.spacing.sm, width: 430 }),
  filterLabel: { width: '92px', marginTop: '6px' },
};

export default ExperimentFilterParams;
