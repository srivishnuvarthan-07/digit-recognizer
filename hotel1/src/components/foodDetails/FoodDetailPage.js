import React from 'react';
import foodItems from 'C:/Users/vivvi/Desktop/hotel/src/food data/foodItems.json';
import './FoodDetailPage.css';

function FoodDetailPage({ foodId, onBack, onAddToOrder }) {
  const food = foodItems.find(item => item.id === foodId);

  if (!food) {
    return (
      <div className="not-found-container">
        <h2>Food item not found</h2>
        <button className="back-to-menu-btn" onClick={onBack}>
          Back to Menu
        </button>
      </div>
    );
  }

  const handleAddToOrder = () => {
    onAddToOrder(food);
    alert(`${food.name} added to order!`);
    onBack();
  };

  return (
    <div>
      {/* Header */}
      <div className="food-detail-header">
        <button onClick={onBack}>← Back</button>
        <h1>Food Details</h1>
      </div>

      {/* Food Detail Content */}
      <div className="food-detail-container">
        {/* Food Image */}
        <div className="food-image">
          <img src={food.image} alt={food.name} />
        </div>

        {/* Food Details */}
        <div className="food-details-card">
          <h2>{food.name}</h2>
          <div>
            <strong>Recipe:</strong>
            {food.recipe}
          </div>
          <div>
            <strong>Price:</strong>
            <span className="food-price">${food.price}</span>
          </div>
          <div>
            <strong>Type:</strong>
            <span className={`food-type ${food.isVeg ? 'veg' : 'non-veg'}`}>
              {food.isVeg ? '🥬 Vegetarian' : '🍖 Non-Vegetarian'}
            </span>
          </div>

          <div style={{ marginTop: '2rem', textAlign: 'center' }}>
            <button
              className="add-to-order-btn"
              onClick={handleAddToOrder}
            >
              Add to Order - ${food.price}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FoodDetailPage;
