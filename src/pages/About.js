import React from 'react';
import '../styles/About.scss';

function About() {
  return (
    <div className="about">
      <h1>
        ABOUT
        I<span className="small-upper">CE</span>
        A<span className="small-upper">NALYTICS</span>
      </h1>
      <section>
        <h2>Our Mission</h2>
        <p>
          IceAnalytics is dedicated to advancing projections, analytics, and quantitative research in professional hockey.
        </p>
        <p>
          We focus on two core objectives. First, to deepen the understanding of player impact, team strategy, and the game of hockey through data-driven analysis. Second, to translate these insights into predictive models that estimate future game outcomes and generate probabilistic projections for players and teams.
        </p>
      </section>
      <section>
        <h2>How It Works</h2>
        <p>IceAnalytics is powered by a sophisticated bootstrapped Monte Carlo ensemble projection engine to generate accurate and probabilistic insights into player and team performance. The following outlines the steps involved in our modeling process:</p>
        <p><strong>1. Data Collection</strong>: We scrape player and team statistical and biographical data from the NHL API for every season since 2007. These analytics provide a rich foundation for the modeling done for this engine. Scraping is done for the concurrent season's data as well, including player statistics, team performance metrics, and the complete game schedule to enable rest-of-season projections.</p>
        <p><strong>2. Model Training</strong>: Leveraging the scraped historical data, our engine trains a series of models to infer metrics such as games played (GP), average time on ice (ATOI), goals (G), primary assists (A1), secondary assits (A2), and many defensive analytics. A variety of model architectures are employed; all of which undergo hyperparameter optimziation via season-fold cross-validation.</p>
        <p><strong>3. Inferencing</strong>: Once the models are trained, we make inferences for the current season by integrating the predictions from these models with each player's ongoing performance.</p>
        <p><strong>4. Bootstrapping and Uncertainty Quantification</strong>: To produce honest probability distributions over each player's projection, we train 500 bootstrapped models per target stat (ATOI, GP rate, and per-60 scoring rates split across even strength, power play, and penalty kill). Each bootstrap holds out a random ~15% of the training data so we can collect its out-of-sample residuals. The per-player projection uncertainty is then decomposed additively: σ²<sub>total</sub>(x) = σ²<sub>epistemic</sub>(x) + σ²<sub>aleatoric</sub>. The <em>epistemic</em> component is the variance of the 500 bootstrap predictions for that specific player — it captures how much uncertainty the model itself has, and is naturally larger for players with sparse or unusual training-data analogues. The <em>aleatoric</em> component is the irreducible noise floor estimated from the pooled out-of-sample residuals across all bootstraps, with the mean epistemic variance subtracted off to avoid double-counting (a deep-ensemble–style decomposition). The resulting per-player standard deviations are then scaled by √(1 − GP / 82) to convert season-total uncertainty into uncertainty over the remaining games, so projections sharpen naturally as the season progresses.</p>
        <p><strong>5. Game Simulations</strong>: We then simulate the remainder of the season by iterating through each scheduled game. Player inferences are adjusted for factors such as teammate and opponent quality, deployment strategies, team dynamics, and home-ice advantage. For each game, we use the players' adjusted inferences to predict win probabilities, projected goal differential, and overtime odds by splitting the game into 30-second chunks. Scoring is modelled probabistically by chunk using a Poisson distribution. As the bootstrapped samples are linearly transformed by the adjusted inferences, this process inherently updates the probability distributions.</p>
        <p><strong>6. Monte Carlo Simulations</strong>: With game probabilities and individual player statistical distributions in hand, we run thousands of Monte Carlo simulations to conduct further probabilistic analysis. This allows us to estimate the likelihood of players reaching certain statistical milestones such as attaining 50 goals, 100 points, or winning the Art Ross trophy. Moreover, this approach enables us to project team outcomes such as playoff chances, Stanley Cup odds, or the probability of securing the first overall draft pick.</p>
        <p>Ultimately, this process renders a model that projects future statistics and probabilities accounting for historical performance, historical peripheral statistics, age, teammate quality, teammate play styles, deployment, opponent quality, among other factors. This approach ensures that every aspect of player and team performance is modeled with both precision and flexibility, accounting for uncertainty and randomness in the game.</p>
      </section>
      <section>
        <h2>Technology Stack</h2>
        <p>
          IceAnalytics runs on a robust and modern technology stack, optimized for scalability and accuracy. Here's a quick overview of what powers our platform:
        </p>
        <ul>
          <li><strong>Backend</strong>: Python</li>
          <li><strong>Database</strong>: Supabase</li>
          <li><strong>Frontend</strong>: ReactJS</li>
          <li><strong>Deployment</strong>: Hosted on Firebase</li>
        </ul>
      </section>
      <section>
        <h2>Contact Us</h2>
        <p>
          We love hearing from our users! Whether you have a question, suggestion, or just want to say hello, feel free to reach out:
        </p>
        <ul>
          <li><strong>Email</strong>: <a href="mailto:hello@iceanalytics.ca">hello@iceanalytics.ca</a></li>
          <li><strong>GitHub</strong>: <a href="https://github.com/andrewderango/IceAnalytics" target="_blank" rel="noopener noreferrer">IceAnalytics GitHub Repository</a></li>
        </ul>
        <p>
          Thank you for visiting IceAnalytics. We hope you enjoy exploring the insights and projections we provide!
        </p>
      </section>
    </div>
  );
}

export default About;