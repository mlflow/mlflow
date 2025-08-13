import { Button, useDesignSystemTheme } from '@databricks/design-system';
import React, { useCallback, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { TracesViewDeleteTraceModal } from './TracesViewDeleteTraceModal';

export const TracesViewControlsActions = ({
  experimentIds,
  rowSelection,
  setRowSelection,
  refreshTraces,
  baseComponentId,
}: {
  experimentIds: string[];
  rowSelection: { [id: string]: boolean };
  setRowSelection: (rowSelection: { [id: string]: boolean }) => void;
  refreshTraces: () => void;
  baseComponentId: string;
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { theme } = useDesignSystemTheme();

  const openModal = useCallback(() => {
    setIsModalOpen(true);
  }, [setIsModalOpen]);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
  }, [setIsModalOpen]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      <Button componentId={`${baseComponentId}.traces_table.delete_traces`} onClick={openModal} danger>
        <FormattedMessage
          defaultMessage="Delete"
          description="Experiment page > traces view controls > Delete button"
        />
      </Button>
      <TracesViewDeleteTraceModal
        experimentIds={experimentIds}
        visible={isModalOpen}
        rowSelection={rowSelection}
        handleClose={closeModal}
        refreshTraces={refreshTraces}
        setRowSelection={setRowSelection}
      />
    </div>
  );
};
