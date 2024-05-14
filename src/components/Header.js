import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.scss';
import logo from '../assets/images/logo.svg';

function Header() {
  const [width, setWidth] = useState(window.innerWidth);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <header className="header">
      <div className="header-name">
        <img src={logo} alt="Logo" />
        <h1>PUCKPROJECTIONS</h1>
      </div>
      {width >= 1000 ? (
        <nav>
          <ul>
            <li><Link to="/home">HOME</Link></li>
            <li><Link to="/games">GAMES</Link></li>
            <li><Link to="/players">PLAYERS</Link></li>
            <li><Link to="/teams">TEAMS</Link></li>
            <li><Link to="/about">ABOUT</Link></li>
            <li><Link to="/">{width}</Link></li>
          </ul>
        </nav>
      ) : (
        <nav>
          <button onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen}>Menu</button>
          {menuOpen && (
            <ul>
              <li><Link to="/home">HOME</Link></li>
              <li><Link to="/games">GAMES</Link></li>
              <li><Link to="/players">PLAYERS</Link></li>
              <li><Link to="/teams">TEAMS</Link></li>
              <li><Link to="/about">ABOUT</Link></li>
              <li><Link to="/">{width}</Link></li>
            </ul>
          )}
        </nav>
      )}
    </header>
  );
}

export default Header;