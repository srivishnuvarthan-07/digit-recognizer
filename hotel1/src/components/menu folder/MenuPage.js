import React from 'react';
import foodItems from 'C:/Users/vivvi/Desktop/hotel/src/food data/foodItems.json';
import './MenuPage.css';

function MenuPage({ onFoodClick, onConfirmOrder, onAddToOrder, orderItems }) {

  const handlePlaceOrder = (e, food) => {
    e.stopPropagation();
    onAddToOrder(food);
    alert(`${food.name} added to order!`);
  };

  return (
    <div>
      {/* Header */}
      <div className="menu-header">
        <h1>Food Menu</h1>
        <button
          onClick={onConfirmOrder}
          className="confirm-order-btn"
        >
          Confirm Order ({orderItems.length})
        </button>
      </div>

      {/* Food Items List */}
      <div className="food-item-container">
        {foodItems.map(food => (
          <div
            key={food.id}
            onClick={() => onFoodClick(food.id)}
            className="food-item"
          >
            <div>
              <h3 className="food-name">{food.name}</h3>
              <span className={`food-tag ${food.isVeg ? 'veg' : 'non-veg'}`}>
                {food.isVeg ? '🥬 Veg' : '🍖 Non-Veg'}
              </span>
            </div>
            <div className="food-actions">
              <span className="food-price">${food.price}</span>
              <button
                onClick={(e) => handlePlaceOrder(e, food)}
                className="place-order-btn"
              >
                Place Order
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MenuPage;
