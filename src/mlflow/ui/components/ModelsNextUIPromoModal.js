import React from 'react';
import './ModelsNextUIPromoModal.css';

const ModelsNextUIPromoModal = () => {
  const [isOpen, setIsOpen] = React.useState(false);

  const handleOpen = () => {
    setIsOpen(true);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <div className="ModelsNextUIPromoModal">
      <button onClick={handleOpen}>Open Modal</button>
      {isOpen && (
        <div className="ModelsNextUIPromoModal-content">
          <h2>Models Next UI Promo Modal</h2>
          <button onClick={handleClose}>Close Modal</button>
        </div>
      )}
    </div>
  );
};

export default ModelsNextUIPromoModal;