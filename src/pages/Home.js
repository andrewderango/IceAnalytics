import React, { useState, useEffect, useRef } from 'react';
import '../styles/Home.scss';
import headshot from '../assets/images/headshot6.png';


function Home() {
  const [isScrolled, setIsScrolled] = useState(false);
  const aboutSectionRef = useRef(null);
  const [hasAnimated, setHasAnimated] = useState(false);

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

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated) {
          const textElements = entry.target.querySelectorAll('h2, p');
          textElements.forEach(element => element.classList.add('reveal'));
          setHasAnimated(true);
        }
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
      }
    );
    if (aboutSectionRef.current) {
      observer.observe(aboutSectionRef.current);
    }
    return () => {
      if (aboutSectionRef.current) {
        observer.unobserve(aboutSectionRef.current);
      }
    };
  }, [hasAnimated]);

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
      <div className="about-section" ref={aboutSectionRef}>
        <h2>ABOUT PUCKPROJECTIONS</h2>
        <div className="underline"></div>
        <p>PuckProjections is a free and open-source NHL simulation engine used to deliver cutting-edge projections and analytics. The platform harnesses the power of ensemble machine learning and Monte Carlo simulations to provide comprehensive insights into NHL games, players, and teams.</p>
        <button onClick={() => window.location.href='/about'}>
          <span>LEARN MORE</span>
        </button>
        {/* <button onClick={() => window.open('https://github.com/andrewderango/NHL-Simulation-Engine', '_blank')}>View Source Code</button> */}
      </div>
      <div className="pages-section">
        <h2>FEATURES</h2>
        <div className="widgets">
          <div className="widget">
            <i className="icon games-icon"></i>
            <hr className="underline" />
            <h3>Games</h3>
            <p>Explore win probabilities and expected goal spreads for upcoming games.</p>
            <button onClick={() => window.location.href='/games'}>
              <span>VIEW GAMES</span>
            </button>
          </div>
          <div className="widget">
            <i className="icon players-icon"></i>
            <hr className="underline" />
            <h3>Players</h3>
            <p>Dive into statistic projections probabilities for your favourite players.</p>
            <button onClick={() => window.location.href='/players'}>
              <span>VIEW PLAYERS</span>
            </button>
          </div>
          <div className="widget">
            <i className="icon teams-icon"></i>
            <hr className="underline" />
            <h3>Teams</h3>
            <p>Get the latest on team performances, playoff probabilties, and Cup odds.</p>
            <button onClick={() => window.location.href='/teams'}>
              <span>VIEW TEAMS</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;