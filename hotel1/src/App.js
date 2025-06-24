import React,{useState} from 'react';
import MenuPage from './components/menu folder/MenuPage.js';
import FoodDetailPage from './components/foodDetails/FoodDetailPage.js';
import OrderPage from './components/order folder/OrderPage.js';

function App() {
  const [currentPage, setCurrentPage] = useState('menu');
  const [selectedFoodId, setSelectedFoodId] = useState(null);
  const [orderItems, setOrderItems] = useState([]);

  const handleFoodClick = (foodId) => {
    setSelectedFoodId(foodId);
    setCurrentPage('detail');
  };

  const handleBackToMenu = () => {
    setCurrentPage('menu');
    setSelectedFoodId(null);
  };

  const handleAddToOrder = (food) => {
    // Check if item already exists in order
    const existingItemIndex = orderItems.findIndex(item => item.id === food.id);
    
    if (existingItemIndex >= 0) {
      // If item exists, increment quantity
      setOrderItems(prev => prev.map((item, index) => 
        index === existingItemIndex 
          ? { ...item, quantity: item.quantity + 1 }
          : item
      ));
    } else {
      // If item doesn't exist, add it with quantity 1
      setOrderItems(prev => [...prev, { ...food, quantity: 1 }]);
    }
  };

  const handleConfirmOrder = () => {
    if (orderItems.length === 0) {
      alert('No items in order!');
      return;
    }
    setCurrentPage('order');
  };

  const handleBackFromOrder = () => {
    setCurrentPage('menu');
  };

  const handleClearOrder = () => {
    setOrderItems([]);
  };

  if (currentPage === 'detail') {
    return (
      <FoodDetailPage 
        foodId={selectedFoodId} 
        onBack={handleBackToMenu}
        onAddToOrder={handleAddToOrder}
      />
    );
  }

  if (currentPage === 'order') {
    return (
      <OrderPage 
        orderItems={orderItems}
        onBack={handleBackFromOrder}
        onClearOrder={handleClearOrder}
      />
    );
  }

  return (
    <MenuPage 
      onFoodClick={handleFoodClick}
      onConfirmOrder={handleConfirmOrder}
      onAddToOrder={handleAddToOrder}
      orderItems={orderItems}
    />
  );
}

export default App;
