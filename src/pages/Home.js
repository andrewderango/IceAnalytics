import React, { useState, useEffect, useRef } from 'react';
import '../styles/Home.scss';
import headshot from '../assets/images/headshot6.png';

function Home() {
  const [isScrolled, setIsScrolled] = useState(false);
  const aboutSectionRef = useRef(null);
  const [aboutHasAnimated, setAboutHasAnimated] = useState(false);
  const widgetRef1 = useRef(null);
  const widgetRef2 = useRef(null);
  const widgetRef3 = useRef(null);
  // const [widgetHasAnimated, setWidgetHasAnimated] = useState(false);

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
        if (entry.isIntersecting && !aboutHasAnimated) {
          const textElements = entry.target.querySelectorAll('h2, p');
          textElements.forEach(element => element.classList.add('reveal'));
          setAboutHasAnimated(true);
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
  }, [aboutHasAnimated]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('slide-up');
          }
        });
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
      }
    );
    if (widgetRef1.current) {
      observer.observe(widgetRef1.current);
    }
    if (widgetRef2.current) {
      observer.observe(widgetRef2.current);
    }
    if (widgetRef3.current) {
      observer.observe(widgetRef3.current);
    }
    return () => {
      if (widgetRef1.current) {
        observer.unobserve(widgetRef1.current);
      }
      if (widgetRef2.current) {
        observer.unobserve(widgetRef2.current);
      }
      if (widgetRef3.current) {
        observer.unobserve(widgetRef3.current);
      }
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
          <div className="widget widget1" ref={widgetRef1}>
            <i className="icon games-icon"></i>
            <hr className="underline" />
            <h3>Games</h3>
            <p>Explore win probabilities and expected goal spreads for upcoming games.</p>
            <button onClick={() => window.location.href='/games'}>VIEW GAMES</button>
          </div>
          <div className="widget widget2" ref={widgetRef2}>
            <i className="icon players-icon"></i>
            <hr className="underline" />
            <h3>Players</h3>
            <p>Dive into statistic projections and probabilities for your favourite players.</p>
            <button onClick={() => window.location.href='/players'}>VIEW PLAYERS</button>
          </div>
          <div className="widget widget3" ref={widgetRef3}>
            <i className="icon teams-icon"></i>
            <hr className="underline" />
            <h3>Teams</h3>
            <p>Get the latest on team performances, playoff probabilties, and Cup odds.</p>
            <button onClick={() => window.location.href='/teams'}>VIEW TEAMS</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;