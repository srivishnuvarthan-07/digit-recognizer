import React, { useState } from 'react';

// Mock data for food items
const foodItems = [
  {
    id: 1,
    name: "Margherita Pizza",
    price: 12.99,
    recipe: "Fresh mozzarella, tomato sauce, basil leaves, olive oil on a crispy thin crust",
    image: "https://images.unsplash.com/photo-1604068549290-dea0e4a305ca?w=400&h=300&fit=crop",
    isVeg: true
  },
  {
    id: 2,
    name: "Chicken Tikka Masala",
    price: 15.99,
    recipe: "Tender chicken pieces in a creamy tomato-based curry sauce with aromatic spices",
    image: "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=400&h=300&fit=crop",
    isVeg: false
  },
  {
    id: 3,
    name: "Caesar Salad",
    price: 9.99,
    recipe: "Crisp romaine lettuce, parmesan cheese, croutons, and classic Caesar dressing",
    image: "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400&h=300&fit=crop",
    isVeg: true
  },
  {
    id: 4,
    name: "Beef Burger",
    price: 13.99,
    recipe: "Juicy beef patty with lettuce, tomato, onion, pickles, and special sauce on a brioche bun",
    image: "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop",
    isVeg: false
  },
  {
    id: 5,
    name: "Vegetable Stir Fry",
    price: 11.99,
    recipe: "Fresh mixed vegetables wok-tossed with garlic, ginger, and soy sauce",
    image: "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&h=300&fit=crop",
    isVeg: true
  },
  {
    id: 6,
    name: "Grilled Salmon",
    price: 18.99,
    recipe: "Fresh Atlantic salmon grilled to perfection with herbs and lemon butter",
    image: "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400&h=300&fit=crop",
    isVeg: false
  },
  {
    id: 7,
    name: "Mushroom Risotto",
    price: 14.99,
    recipe: "Creamy arborio rice with wild mushrooms, parmesan, and truffle oil",
    image: "https://images.unsplash.com/photo-1476124369491-e7addf5db371?w=400&h=300&fit=crop",
    isVeg: true
  },
  {
    id: 8,
    name: "Fish Tacos",
    price: 12.99,
    recipe: "Grilled white fish with cabbage slaw, pico de gallo, and chipotle crema",
    image: "https://images.unsplash.com/photo-1565299585323-38174c4a6234?w=400&h=300&fit=crop",
    isVeg: false
  }
];

// Menu Page Component
function MenuPage({ onFoodClick, onConfirmOrder, orderCount }) {
  const [orders, setOrders] = useState([]);

  const handlePlaceOrder = (e, food) => {
    e.stopPropagation();
    setOrders(prev => [...prev, food]);
    alert(`${food.name} added to order!`);
  };

  const handleConfirmOrder = () => {
    if (orders.length === 0) {
      alert('No items in order!');
      return;
    }
    const total = orders.reduce((sum, item) => sum + item.price, 0);
    alert(`Order confirmed! Total: $${total.toFixed(2)}\nItems: ${orders.map(item => item.name).join(', ')}`);
    setOrders([]);
    onConfirmOrder();
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f5f5f5',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        position: 'sticky',
        top: 0,
        backgroundColor: '#fff',
        padding: '1rem',
        borderBottom: '1px solid #ddd',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        zIndex: 100
      }}>
        <h1 style={{ margin: 0, color: '#333' }}>Food Menu</h1>
        <button
          onClick={handleConfirmOrder}
          style={{
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#218838'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#28a745'}
        >
          Confirm Order ({orders.length})
        </button>
      </div>

      {/* Food Items List */}
      <div style={{ padding: '1rem' }}>
        {foodItems.map(food => (
          <div
            key={food.id}
            onClick={() => onFoodClick(food.id)}
            style={{
              width: '100%',
              backgroundColor: '#fff',
              border: '1px solid #ddd',
              borderRadius: '12px',
              padding: '1rem',
              marginBottom: '1rem',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              transition: 'transform 0.2s, box-shadow 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            }}
          >
            <div>
              <h3 style={{ margin: '0 0 0.5rem 0', color: '#333' }}>{food.name}</h3>
              <span style={{
                display: 'inline-block',
                padding: '0.25rem 0.5rem',
                borderRadius: '12px',
                fontSize: '0.8rem',
                backgroundColor: food.isVeg ? '#d4edda' : '#f8d7da',
                color: food.isVeg ? '#155724' : '#721c24'
              }}>
                {food.isVeg ? '🥬 Veg' : '🍖 Non-Veg'}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#333' }}>
                ${food.price}
              </span>
              <button
                onClick={(e) => handlePlaceOrder(e, food)}
                style={{
                  backgroundColor: '#007bff',
                  color: 'white',
                  border: 'none',
                  padding: '0.5rem 1rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.backgroundColor = '#0056b3'}
                onMouseLeave={(e) => e.target.style.backgroundColor = '#007bff'}
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

// Food Detail Page Component
function FoodDetailPage({ foodId, onBack }) {
  const food = foodItems.find(item => item.id === foodId);

  if (!food) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center',
        minHeight: '100vh',
        backgroundColor: '#f5f5f5'
      }}>
        <h2>Food item not found</h2>
        <button 
          onClick={onBack}
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          Back to Menu
        </button>
      </div>
    );
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f5f5f5',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        backgroundColor: '#fff',
        padding: '1rem',
        borderBottom: '1px solid #ddd',
        display: 'flex',
        alignItems: 'center'
      }}>
        <button
          onClick={onBack}
          style={{
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            cursor: 'pointer',
            marginRight: '1rem',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#545b62'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#6c757d'}
        >
          ← Back
        </button>
        <h1 style={{ margin: 0, color: '#333' }}>Food Details</h1>
      </div>

      {/* Food Detail Content */}
      <div style={{ 
        padding: '2rem',
        maxWidth: '600px',
        margin: '0 auto'
      }}>
        {/* Food Image */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <img
            src={food.image}
            alt={food.name}
            style={{
              width: '100%',
              maxWidth: '400px',
              height: '300px',
              objectFit: 'cover',
              borderRadius: '12px',
              boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
            }}
          />
        </div>

        {/* Food Details */}
        <div style={{
          backgroundColor: '#fff',
          padding: '2rem',
          borderRadius: '12px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ 
            margin: '0 0 1rem 0', 
            color: '#333',
            textAlign: 'center'
          }}>
            {food.name}
          </h2>
          
          <div style={{ lineHeight: '1.8', color: '#555' }}>
            <div style={{ marginBottom: '1rem' }}>
              <strong>Recipe:</strong><br />
              {food.recipe}
            </div>
            
            <div style={{ marginBottom: '1rem' }}>
              <strong>Price:</strong><br />
              <span style={{ fontSize: '1.5rem', color: '#28a745', fontWeight: 'bold' }}>
                ${food.price}
              </span>
            </div>
            
            <div>
              <strong>Type:</strong><br />
              <span style={{
                display: 'inline-block',
                padding: '0.5rem 1rem',
                borderRadius: '20px',
                fontSize: '1rem',
                backgroundColor: food.isVeg ? '#d4edda' : '#f8d7da',
                color: food.isVeg ? '#155724' : '#721c24',
                fontWeight: 'bold'
              }}>
                {food.isVeg ? '🥬 Vegetarian' : '🍖 Non-Vegetarian'}
              </span>
            </div>
          </div>

          <div style={{ 
            marginTop: '2rem',
            textAlign: 'center' 
          }}>
            <button
              onClick={() => {
                alert(`${food.name} added to order!`);
                onBack();
              }}
              style={{
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                padding: '1rem 2rem',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '1.1rem',
                fontWeight: 'bold',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#218838'}
              onMouseLeave={(e) => e.target.style.backgroundColor = '#28a745'}
            >
              Add to Order - ${food.price}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main App Component


export default MenuPage;
// export default MenuPa