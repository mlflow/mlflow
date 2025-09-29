const DeleteRunModal: React.FC<
  React.PropsWithChildren<{
    isOpen?: boolean;
    onClose?: () => void;
    selectedRunIds?: string[];
  }>
>;

export default DeleteRunModal;
