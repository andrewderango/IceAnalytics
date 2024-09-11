import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
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
        <h1>
          I<span style={{ fontSize: 'smaller' }}>CE</span>
          A<span style={{ fontSize: 'smaller' }}>NALYTICS</span>
        </h1>
      </div>
      {width >= 1000 ? (
        <nav>
          <ul>
            <li><NavLink to="/home" activeStyle={{ color: 'goldenrod' }}>HOME</NavLink></li>
            <li><NavLink to="/games" activeStyle={{ color: 'goldenrod' }}>GAMES</NavLink></li>
            <li><NavLink to="/players" activeStyle={{ color: 'goldenrod' }}>PLAYERS</NavLink></li>
            <li><NavLink to="/teams" activeStyle={{ color: 'goldenrod' }}>TEAMS</NavLink></li>
            <li><NavLink to="/about" activeStyle={{ color: 'goldenrod' }}>ABOUT</NavLink></li>
            <li><NavLink to="/badurl" activeStyle={{ color: 'goldenrod' }}>{width}x{height}</NavLink></li>
          </ul>
        </nav>
      ) : (
        <nav ref={menuRef}>
          <button onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen}>
            <i className="fas fa-bars"></i>
          </button>
          {menuOpen && (
            <ul>
              <li><NavLink to="/home" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>HOME</NavLink></li>
              <li><NavLink to="/games" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>GAMES</NavLink></li>
              <li><NavLink to="/players" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>PLAYERS</NavLink></li>
              <li><NavLink to="/teams" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>TEAMS</NavLink></li>
              <li><NavLink to="/about" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>ABOUT</NavLink></li>
              <li><NavLink to="/badurl" activeStyle={{ color: 'goldenrod' }} onClick={() => setMenuOpen(false)}>{width}x{height}</NavLink></li>
            </ul>
          )}
        </nav>
      )}
    </header>
  );
}

export default Header;
