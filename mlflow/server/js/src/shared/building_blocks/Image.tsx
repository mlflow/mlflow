import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CloseIcon,
  DesignSystemContext,
  RedoIcon,
  UndoIcon,
  ZoomInIcon,
  ZoomOutIcon,
} from '@databricks/design-system';
import { useContext } from 'react';
import RcImage from 'rc-image';
import './Image.css';
import { MLflowImagePreviewContainer } from '../../common/components/DesignSystemContainer';

const icons = {
  rotateLeft: <UndoIcon />,
  rotateRight: <RedoIcon />,
  zoomIn: <ZoomInIcon />,
  zoomOut: <ZoomOutIcon />,
  close: <CloseIcon />,
  left: <ArrowLeftIcon />,
  right: <ArrowRightIcon />,
};

export const ImagePreviewGroup = ({
  children,
  visible,
  onVisibleChange,
}: {
  children: React.ReactNode;
  visible: boolean;
  onVisibleChange: (v: boolean) => void;
}) => {
  const { getImagePreviewPopupContainer } = useContext(MLflowImagePreviewContainer);

  return (
    <RcImage.PreviewGroup
      icons={icons}
      preview={{
        visible: visible,
        getContainer: getImagePreviewPopupContainer,
        onVisibleChange: (v) => onVisibleChange(v),
      }}
    >
      {children}
    </RcImage.PreviewGroup>
  );
};

export { RcImage as Image };
