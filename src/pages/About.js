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
          At IceAnalytics, our goal is to revolutionize how NHL performance is understood, analyzed, and predicted. By providing a comprehensive, open-source framework for data-driven insights into NHL games, players, and teams, we hope to empower fans and analysts with a deeper understanding of the game.
        </p>
      </section>
      <section>
        <h2>How It Works</h2>
        <p>IceAnalytics is powered by a sophisticated bootstrapped Monte Carlo ensemble projection engine to generate accurate and probabilistic insights into player and team performance. The following outlines the steps involved in our modeling process:</p>
        <p><strong>1. Data Collection</strong>: We scrape player and team statistical and biographical data from the NHL API for every season since 2007, widely considered to be the start of the analytics era in the NHL. The 2006-07 NHL season was the first to have publicized shot data became available and enable advanced models like expected goals (xG). These advanced analytics data provide a rich foundation for the modeling done for this engine. Scraping is done for the concurrent season's data as well, including player statistics, team performance metrics, and the complete game schedule to enable rest-of-season projections.</p>
        <p><strong>2. Model Training</strong>: Leveraging the scraped historical data, our engine trains a series of models to infer metrics such as games played (GP), average time on ice (ATOI), goals (G), primary assists (A1), secondary assits (A2), and many defensive analytics such as team-level goals against (GA). These models include support vector regression, neural networks, and XGBoost, whose hyperparameters were optimized via Optuna.</p>
        <p><strong>3. Inferencing</strong>: Once the models are trained, we make inferences for the current season by integrating the predictions from these models with each player's ongoing performance. Subsequently, a Savitzky-Golay filter is applied for calibrative scaling. This helps the inferences better align with historical scaling and ensure a more accurate trend in player and team performance from season to season.</p>
        <p><strong>4. Bootstrapping and PRUS</strong>: To account for variability and uncertainty, we train 500 bootstrapped models for each key statistic: ATOI, G, A1, A2 and GP. Prior to the bootstrap sampling, the data is split into a test and train dataset, enabling us to compute the variance of the bootstrap models' residuals in parallel. This idea fundamentally underpins a novel algorithm that we are calling the Parallelized Residual-based Uncertainty Scaling (PRUS) algorithm, which is critical in accounting for epistemic uncertainty. If we scale the variance of the predictions across the 500 bootstraps for a specific player by the computed residual-based variances, we can estimate the total uncertainty of the player's performance and effectively generate a probability distribution for each of the key target variables. This approach captures both <em>epistemic</em> and <em>aleatoric</em> uncertainty, while dynamically adjusting based on the player's current-season performance and its progression.</p>
        <p><strong>5. Game Simulations</strong>: We then simulate the remainder of the season by looping through each scheduled game. Player inferences are adjusted for factors such as teammate and opponent quality, deployment strategies, team dynamics, and home-ice advantage. For each game, we use the players' adjusted inferences to predict win probabilities, projected goal differential, and overtime odds by splitting the game into 30-second chunks and using a Poisson distribution for scoring. As the bootstrapped samples mask the adjusted inferences, this process inherently updates the probability distributions.</p>
        <p><strong>6. Monte Carlo Simulations</strong>: With game probabilities and individual player statistical distributions in hand, we run thousands of Monte Carlo simulations to conduct further probabilistic analysis. This allows us to estimate the likelihood of players reaching certain statistical milestones such as attaining 50 goals, 100 points, or winning the Art Ross trophy. Moreover, this approach enables us to project team outcomes such as playoff chances, Stanley Cup odds, or the probability of securing the first overall draft pick.</p>
        <p>Ultimately, this process renders a model that projects future statistics and probabilities accounting for historical performance, historical peripheral statistics, age, teammate quality, teammate play styles, deployment, opponent quality, among other factors. This approach ensures that every aspect of player and team performance is modeled with both precision and flexibility, accounting for uncertainty and randomness in the game.</p>
      </section>
      <section>
        <h2>Technology Stack</h2>
        <p>
          IceAnalytics runs on a robust and modern technology stack, optimized for scalability and accuracy. Here's a quick overview of what powers our platform:
        </p>
        <ul>
          <li><strong>Backend</strong>: At the heart of our analysis is Python, which powers our projection and analytics models. Leveraging robust libraries such as SciPy, scikit-learn, XGBoost and TensorFlow, Python executes the entire projection engine.</li>
          <li><strong>Database</strong>: Our data is stored and served using Supabase for efficient and secure database management.</li>
          <li><strong>Frontend</strong>: The website is built using ReactJS and styled with JSX and SCSS.</li>
          <li><strong>Deployment</strong>: The site is deployed using Firebase, ensuring high availability and fast load times.</li>
        </ul>
      </section>
      <section>
        <h2>Contact Us</h2>
        <p>
          We love hearing from our users! Whether you have a question, suggestion, or just want to say hello, feel free to reach out:
        </p>
        <ul>
          <li><strong>Email</strong>: <a href="mailto:contact@iceanalytics.ca">contact@iceanalytics.ca</a></li>
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