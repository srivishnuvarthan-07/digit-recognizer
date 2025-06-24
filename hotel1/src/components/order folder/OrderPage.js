import React, { useState } from 'react';
import './OrderPage.css';

const OrderPage = ({ orderItems = [], onBack, onClearOrder }) => {
    
  const [isOnlinePayment, setIsOnlinePayment] = useState(false);
  const [hasInteractedWithToggle, setHasInteractedWithToggle] = useState(false);

  // Calculate total cost
  const calculateTotal = () => {
    return orderItems.reduce((total, item) => {
      return total + (item.price * item.quantity);
    }, 0).toFixed(2);
  };

  const handlePaymentModeToggle = () => {
    setIsOnlinePayment(!isOnlinePayment);
    setHasInteractedWithToggle(true);
  };

  const handlePayClick = () => {
    if (isOnlinePayment) {
      console.log('Processing online payment...');
      alert(`Online payment processed successfully!\nTotal: $${calculateTotal()}\nOrder items: ${orderItems.map(item => `${item.name} (x${item.quantity})`).join(', ')}`);
    } else {
      console.log('Processing offline payment...');
      alert(`Offline payment processed successfully!\nTotal: $${calculateTotal()}\nOrder items: ${orderItems.map(item => `${item.name} (x${item.quantity})`).join(', ')}`);
    }
    
    // Clear the order after successful payment
    onClearOrder();
    onBack();
  };

  return (
    <div className="order-page">
      <div className="order-header">
        <h1>Your Order</h1>
        {onBack && (
          <button className="back-button" onClick={onBack}>
            ← Back
          </button>
        )}
      </div>

      <div className="order-content">
        {orderItems.length === 0 ? (
          <div className="empty-order">
            <p>No items in your order yet.</p>
          </div>
        ) : (
          <>
            <div className="order-items-container">
              {orderItems.map((item, index) => (
                <div key={`${item.id}-${index}`} className="order-item">
                  <div className="item-info">
                    <h3 className="item-name">{item.name}</h3>
                    <div className="item-details">
                      <span className="item-quantity">Qty: {item.quantity}</span>
                      <span className="item-price">${item.price.toFixed(2)} each</span>
                    </div>
                  </div>
                  <div className="item-total">
                    ${(item.price * item.quantity).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>

            <div className="order-summary">
              <div className="total-cost">
                <h2>Total: ${calculateTotal()}</h2>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="order-footer">
        <div className="payment-mode-container">
          <label className="payment-toggle">
            <span className="toggle-label">Payment Mode:</span>
            <div className="slider-container">
              <span className={`mode-label ${!isOnlinePayment ? 'active' : ''}`}>
                Offline
              </span>
              <div className="slider-wrapper">
                <input
                  type="checkbox"
                  className="slider-input"
                  checked={isOnlinePayment}
                  onChange={handlePaymentModeToggle}
                />
                <span className="slider"></span>
              </div>
              <span className={`mode-label ${isOnlinePayment ? 'active' : ''}`}>
                Online
              </span>
            </div>
          </label>
        </div>

        {orderItems.length > 0 && (
          <button 
            className="pay-button"
            onClick={handlePayClick}
            disabled={orderItems.length === 0 || !hasInteractedWithToggle}
          >
            Pay ${calculateTotal()}
          </button>
        )}
      </div>
    </div>
  );
};

export default OrderPage;