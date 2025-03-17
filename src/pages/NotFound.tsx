import React from 'react';
import notFound from '../assets/images/404.png';
import '../styles/NotFound.scss';

function NotFound() {
  return (
    <div className="not-found">
        <img src={notFound} alt="404"/>
        <h1>404</h1>
        <p>Page not found</p>
    </div>
  );
}

export default NotFound;