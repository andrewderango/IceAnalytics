import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.scss';
import logo from '../assets/images/logo.svg';

function Header() {
  const [width, setWidth] = useState(window.innerWidth);
  const [height, setHeight] = useState(window.innerHeight);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef();

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth);
      setHeight(window.innerHeight);
    };
  
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    };

    const handleScroll = () => {
      setMenuOpen(false);
    };

    document.addEventListener('mousedown', handleClickOutside);
    window.addEventListener('scroll', handleScroll);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [menuRef]);

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
            <li><Link to="/">{width}x{height}</Link></li>
          </ul>
        </nav>
      ) : (
        <nav ref={menuRef}>
          <button onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen}>
            <i className="fas fa-bars"></i>
          </button>
          {menuOpen && (
            <ul>
              <li><Link to="/home" onClick={() => setMenuOpen(false)}>HOME</Link></li>
              <li><Link to="/games" onClick={() => setMenuOpen(false)}>GAMES</Link></li>
              <li><Link to="/players" onClick={() => setMenuOpen(false)}>PLAYERS</Link></li>
              <li><Link to="/teams" onClick={() => setMenuOpen(false)}>TEAMS</Link></li>
              <li><Link to="/about" onClick={() => setMenuOpen(false)}>ABOUT</Link></li>
              <li><Link to="/" onClick={() => setMenuOpen(false)}>{width}x{height}</Link></li>
            </ul>
          )}
        </nav>
      )}
    </header>
  );
}

export default Header;