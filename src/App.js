import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './pages/Home';
import Games from './pages/Games';
import Players from './pages/Players';
import Teams from './pages/Teams';
import About from './pages/About';
import Header from './components/Header';
import Footer from './components/Footer';

function App() {
  return (
    <Router>
      <Header />
      <Switch>
        <Route exact path="/home" component={Home} />
        <Route exact path="/games" component={Games} />
        <Route path="/players" component={Players} />
        <Route path="/teams" component={Teams} />
        <Route path="/about" component={About} />
      </Switch>
      <Footer />
    </Router>
  );
}

export default App;
