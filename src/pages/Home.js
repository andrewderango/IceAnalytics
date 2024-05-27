import React from 'react';
import '../styles/Home.scss';
import headshot from '../assets/images/headshot6.png';

function Home() {
  return (
    <div className="home">
      <div className="text-container">
        <h1>PUCKPROJECTIONS</h1>
        <div className="underline"></div>
        <p>Transformative NHL projections and analytics powered by machine learning</p>
      </div>
      <img src={headshot} alt="Headshot" className="headshot" />
    </div>
  );
}

export default Home;