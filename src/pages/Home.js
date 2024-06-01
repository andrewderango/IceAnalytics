import React, { useState, useEffect } from 'react';
import '../styles/Home.scss';
import headshot from '../assets/images/headshot6.png';


function Home() {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      const scrollPos = window.scrollY || window.scrollTop || document.getElementsByTagName("html")[0].scrollTop;
      setIsScrolled(scrollPos > 0 ? true : false);
    };

    window.addEventListener("scroll", onScroll);

    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  return (
    <div className="home">
      <div className="landing-section">
        <div className="text-container slide-in-left">
          <h1 className="slide-in-h1">PUCKPROJECTIONS</h1>
          <div className="underline"></div>
          <p>Transformative NHL projections and analytics powered by machine learning</p>
        </div>
        <img src={headshot} alt="Headshot" className="headshot slide-in-right" />
        {!isScrolled && <div id="scroll-down-arrow"></div>}
      </div>
      <div className="about-section">
        <h2>ABOUT PUCKPROJECTIONS</h2>
        <div className="underline"></div>
        <p>PuckProjections is a free and open-source NHL simulation engine used to deliver cutting-edge projections and analytics. The platform harnesses the power of ensemble machine learning and Monte Carlo simulations to provide comprehensive insights into NHL games, players, and teams.</p>
        <button onClick={() => window.location.href='/about'}>
          <span>LEARN MORE</span>
        </button>
        {/* <button onClick={() => window.open('https://github.com/andrewderango/NHL-Simulation-Engine', '_blank')}>View Source Code</button> */}
      </div>
      <div className="scroll-down-text">Scroll down</div> {/* temp */}
    </div>
  );
}

export default Home;