  const loadedRecordsCount = datasetRecords.length;
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const selectedRowIds = Object.keys(rowSelection).filter((key) => rowSelection[key]);
  const selectedCount = selectedRowIds.length;
  const hasSelection = selectedCount > 0;

  const { deleteDatasetRecordsMutation, isLoading: isDeleting } = useDeleteDatasetRecordsMutation({
    onSuccess: () => {
      setRowSelection({});
      setShowDeleteConfirm(false);
      // TODO: optional success toast
    },
    onError: (error) => {
      setShowDeleteConfirm(false);
      // TODO: optional error toast
      // eslint-disable-next-line no-console
      console.error('Failed to delete records:', error);
    },
  });

  const handleDeleteClick = () => {
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = () => {
    if (!hasSelection || !datasetId) {
      return;
    }

    deleteDatasetRecordsMutation({
      datasetId,
      datasetRecordIds: selectedRowIds,
    });
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            paddingLeft: theme.spacing.sm,
            paddingRight: theme.spacing.sm,
          }}
        >
          <Typography.Title level={3} withoutMargins>
            {datasetName}
          </Typography.Title>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="Displaying {loadedRecordsCount} of {totalRecordsCount, plural, =1 {1 record} other {# records}}"
              description="Label for the number of records displayed"
              values={{ loadedRecordsCount: loadedRecordsCount ?? 0, totalRecordsCount: totalRecordsCount ?? 0 }}
            />
          </Typography.Text>
        </div>
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.xs }}>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button componentId="mlflow.eval-datasets.records-toolbar.row-size-toggle" icon={<RowsIcon />} />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              <DropdownMenu.RadioGroup
                componentId="mlflow.eval-datasets.records-toolbar.row-size-radio"
                value={rowSize}
                onValueChange={(value) => setRowSize(value as 'sm' | 'md' | 'lg')}
              >
                <DropdownMenu.Label>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Row height" description="Label for the row height radio group" />
                  </Typography.Text>
                </DropdownMenu.Label>
                <DropdownMenu.RadioItem key="sm" value="sm">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Small" description="Small row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
                <DropdownMenu.RadioItem key="md" value="md">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Medium" description="Medium row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
                <DropdownMenu.RadioItem key="lg" value="lg">
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>
                    <FormattedMessage defaultMessage="Large" description="Large row size" />
                  </Typography.Text>
                </DropdownMenu.RadioItem>
              </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button componentId="mlflow.eval-datasets.records-toolbar.columns-toggle" icon={<ColumnsIcon />} />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {columns.map((column) => (
                <DropdownMenu.CheckboxItem
                  componentId="mlflow.eval-datasets.records-toolbar.column-checkbox"
                  key={column.id}
                  checked={columnVisibility[column.id ?? ''] ?? false}
                  onCheckedChange={(checked) =>
                    setColumnVisibility({
                      ...columnVisibility,
                      [column.id ?? '']: checked,
                    })
                  }
                >
                  <DropdownMenu.ItemIndicator />
                  <Typography.Text>{column.header}</Typography.Text>
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.Content>
          </DropdownMenu.Root>

          {hasSelection && (
            <Button
              componentId="mlflow.eval-datasets.records-toolbar.delete-button"
              icon={<TrashIcon />}
              onClick={handleDeleteClick}
              disabled={isDeleting}
              danger
            >
              <FormattedMessage
                defaultMessage="Delete ({selectedCount})"
                description="Delete selected records button"
                values={{ selectedCount }}
              />
            </Button>
          )}
        </div>
      </div>
      <div
        css={{
          paddingLeft: theme.spacing.sm,
          paddingRight: theme.spacing.sm,
        }}
      >
        <Input
          componentId="mlflow.eval-datasets.records-toolbar.search-input"
          prefix={<SearchIcon />}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search inputs and expectations',
            description: 'Placeholder for the evaluation dataset records search input',
          })}
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          css={{ maxWidth: '540px', flex: 1 }}
        />
      </div>

      {showDeleteConfirm && (
        <Dialog.Root open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>
                <FormattedMessage
                  defaultMessage="Delete Records"
                  description="Delete records confirmation dialog title"
                />
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Are you sure you want to delete {selectedCount, plural, =1 {1 record} other {# records}}? This action cannot be undone."
                  description="Delete records confirmation message"
                  values={{ selectedCount }}
                />
              </Typography.Text>
            </Dialog.Body>
            <Dialog.Footer>
              <Button onClick={handleDeleteCancel} disabled={isDeleting}>
                <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
              </Button>
              <Button onClick={handleDeleteConfirm} disabled={isDeleting} danger>
                <FormattedMessage defaultMessage="Delete" description="Delete button" />
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Root>
      )}
    </div>
  );
};