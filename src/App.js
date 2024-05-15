import React from 'react';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
import Home from './pages/Home';
import Games from './pages/Games';
import Players from './pages/Players';
import Teams from './pages/Teams';
import About from './pages/About';
import Header from './components/Header';
import Footer from './components/Footer';
import NotFound from './pages/NotFound';

function App() {
  return (
    <Router>
      <Header />
      <Switch>
        <Route path="/home" component={Home} />
        <Route path="/games" component={Games} />
        <Route path="/players" component={Players} />
        <Route path="/teams" component={Teams} />
        <Route path="/about" component={About} />
        <Route exact path="/" render={() => <Redirect to="/home" />} />
        <Route component={NotFound} /> {/* This will be rendered if no other Route matches */}
      </Switch>
      <Footer />
    </Router>
  );
}

export default App;