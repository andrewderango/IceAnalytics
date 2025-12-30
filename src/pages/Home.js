import React, { useState, useEffect, useRef } from 'react';
import '../styles/Home.scss';
import headshot from '../assets/images/headshot6.png';

function Home() {
  const [isScrolled, setIsScrolled] = useState(false);
  const heroSectionRef = useRef(null);
  const statsRef = useRef(null);
  const featuresRef = useRef(null);
  const widgetRefs = [useRef(null), useRef(null), useRef(null)];

  useEffect(() => {
    const onScroll = () => {
      const scrollPos = window.scrollY;
      setIsScrolled(scrollPos > 50);
    };

    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-view');
          }
        });
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
      }
    );

    const elements = [statsRef.current, featuresRef.current, ...widgetRefs.map(ref => ref.current)];
    elements.forEach(el => el && observer.observe(el));

    return () => {
      elements.forEach(el => el && observer.unobserve(el));
    };
  }, []);

  const stats = [
    { value: "50+", label: "Predictive Models" },
    { value: "800+", label: "Player Projections" },
    { value: "65M+", label: "Daily Game Simulations" }
  ];

  const features = [
    {
      icon: "games-icon",
      title: "Game Intelligence",
      description: "Game-level win probabilities, scoring distributions, and player stat predictions.",
      action: "Explore Games",
      link: "/games"
    },
    {
      icon: "players-icon",
      title: "Player Analytics",
      description: "Projections for individual player statistics with outcome uncertainty distributions.",
      action: "View Players",
      link: "/players"
    },
    {
      icon: "teams-icon",
      title: "Team Insights",
      description: "Team season projections, strength metrics, and playoff qualification probabilities.",
      action: "Analyze Teams",
      link: "/teams"
    }
  ];

  return (
    <div className="home">
      <div className="hero-section" ref={heroSectionRef}>
        <div className="hero-overlay"></div>
        <div className="hero-content">
          <div className="hero-text">
            <div className="hero-badge">HOCKEY ANALYTICS ENGINE</div>
            <h1 className="hero-title">
              <span className="title-accent">ICE</span>ANALYTICS
            </h1>
            <p className="hero-subtitle">
              Data driven projections and analytical insights for the NHL powered by simulation based modeling and large scale quantitative analysis
            </p>
            <div className="hero-actions">
              <button className="btn-primary" onClick={() => window.location.href='/games'}>
                Get Started
              </button>
              <button className="btn-secondary" onClick={() => window.location.href='/about'}>
                Learn More
              </button>
            </div>
          </div>
          <div className="hero-image">
            <img src={headshot} alt="NHL Analytics Visualization" className="headshot" />
          </div>
        </div>
        <div className={`scroll-indicator ${isScrolled ? 'hidden' : ''}`}>
          <span>Scroll to explore</span>
          <div className="scroll-arrow"></div>
        </div>
      </div>

      <section className="stats-section" ref={statsRef}>
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <div key={index} className="stat-card">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="features-section" ref={featuresRef}>
        <div className="section-header">
          <span className="section-tag">CAPABILITIES</span>
          <h2 className="section-title">Advanced Analytics Suite</h2>
          <p className="section-description">
            A comprehensive set of quantitative tools for analyzing NHL games, players, and teams through simulation and predictive modeling
          </p>
        </div>
        
        <div className="features-grid">
          {features.map((feature, index) => (
            <div 
              key={index} 
              className="feature-card" 
              ref={widgetRefs[index]}
            >
              <div className="feature-icon-wrapper">
                <i className={`icon ${feature.icon}`}></i>
              </div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
              <button 
                className="feature-button"
                onClick={() => window.location.href=feature.link}
              >
                {feature.action}
                <span className="arrow">→</span>
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-section">
        <div className="cta-content">
          <h2 className="cta-title">Experience the Future of Hockey Analytics</h2>
          <p className="cta-description">
            Explore player, team, and game projections powered by our predictive modeling engine. Open-source, transparent, and built for enthusiasts.
          </p>
          <div className="cta-actions">
            <button className="btn-primary" onClick={() => window.location.href='/games'}>
              Start Analyzing
            </button>
            <button 
              className="btn-outline"
              onClick={() => window.open('https://github.com/andrewderango/IceAnalytics', '_blank')}
            >
              View on GitHub
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;