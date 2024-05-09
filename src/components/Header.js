import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.scss';
import logo from '../assets/logo.svg';

function Header() {
  return (
    <header className="header">
      <div className="header-name">
        <img src={logo} alt="Logo" />
        <h1>PUCKPROJECTIONS</h1>
      </div>
      <nav>
        <ul>
          <li><Link to="/home">HOME</Link></li>
          <li><Link to="/games">GAMES</Link></li>
          <li><Link to="/players">PLAYERS</Link></li>
          <li><Link to="/teams">TEAMS</Link></li>
          <li><Link to="/about">ABOUT</Link></li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;