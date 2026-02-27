const RestoreRunModal: React.FC<
  React.PropsWithChildren<{
    isOpen?: boolean;
    onClose?: () => void;
    selectedRunIds?: string[];
  }>
>;

export default RestoreRunModal;
