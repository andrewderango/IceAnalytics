import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.scss';

function Header() {
  return (
    <header style={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      padding: '10px', 
      position: 'fixed', 
      top: '0', 
      width: '100%', 
      background: '#333',
      zIndex: '1000'
    }}>
      <h1 style={{ margin: '0' }}>PuckProjections</h1>
      <nav>
        <ul style={{ display: 'flex', gap: '10px', listStyle: 'none', margin: '0' }}>
          <li><Link to="/">Home</Link></li>
          <li><Link to="/players">Players</Link></li>
          <li><Link to="/teams">Teams</Link></li>
          <li><Link to="/about">About</Link></li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;