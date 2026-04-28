# IceAnalytics

[![GitHub release](https://img.shields.io/github/v/release/andrewderango/IceAnalytics.svg)](https://github.com/andrewderango/IceAnalytics/releases)
[![Deploy to Firebase Hosting on PR](https://github.com/andrewderango/IceAnalytics/actions/workflows/firebase-hosting-pull-request.yml/badge.svg)](https://github.com/andrewderango/IceAnalytics/actions/workflows/firebase-hosting-pull-request.yml)
[![Deploy to Firebase Hosting on merge](https://github.com/andrewderango/IceAnalytics/actions/workflows/firebase-hosting-merge.yml/badge.svg)](https://github.com/andrewderango/IceAnalytics/actions/workflows/firebase-hosting-merge.yml)
[![License](https://img.shields.io/badge/license-GPLv3.0-blue.svg)](https://opensource.org/license/gpl-3-0)

IceAnalytics is an open-source NHL analytics and projections platform that provides comprehensive data-driven insights into player and team performance, game outcomes, and playoff probabilities. Powered by a bootstrapped Monte Carlo ensemble projection engine, IceAnalytics offers a unique and in-depth look at the NHL, combining advanced analytics with user-friendly access to predictions. Visit the site at [iceanalytics.ca](https://iceanalytics.ca)!

## Mission

At IceAnalytics, our goal is to revolutionize how NHL performance is understood, analyzed, and predicted. By providing a comprehensive, open-source framework for data-driven insights into NHL games, players, and teams, we hope to empower fans and analysts with a deeper understanding of the game.

## How It Works

IceAnalytics is powered by a sophisticated bootstrapped Monte Carlo ensemble projection engine to generate accurate and probabilistic insights into player and team performance. The following outlines the steps involved in our modeling process:

1. **Data Collection**: We scrape player and team statistical and biographical data from the NHL API for every season since 2007. These analytics provide a rich foundation for the modeling done for this engine. Scraping is done for the concurrent season's data as well, including player statistics, team performance metrics, and the complete game schedule to enable rest-of-season projections.

2. **Model Training**: Leveraging the scraped historical data, our engine trains a series of models to infer metrics such as games played (GP), average time on ice (ATOI), goals (G), primary assists (A1), secondary assits (A2), and many defensive analytics. A variety of model architectures are employed; all of which undergo hyperparameter optimziation via season-fold cross-validation.

3. **Inferencing**: Once the models are trained, we make inferences for the current season by integrating the predictions from these models with each player's ongoing performance.

4. **Bootstrapping and Uncertainty Quantification**: To produce honest probability distributions over each player's projection, we train 500 bootstrapped models per target stat (ATOI, GP rate, and per-60 scoring rates split across even strength, power play, and penalty kill). Each bootstrap holds out a random ~15% of the training data so we can collect its out-of-sample residuals. The per-player projection uncertainty is then decomposed additively: `σ²_total(x) = σ²_epistemic(x) + σ²_aleatoric`. The *epistemic* component is the variance of the 500 bootstrap predictions for that specific player — it captures how much uncertainty the model itself has, and is naturally larger for players with sparse or unusual training-data analogues. The *aleatoric* component is the irreducible noise floor estimated from the pooled out-of-sample residuals across all bootstraps, with the mean epistemic variance subtracted off to avoid double-counting (a deep-ensemble–style decomposition). The resulting per-player standard deviations are then scaled by `√(1 − GP / 82)` to convert season-total uncertainty into uncertainty over the remaining games, so projections sharpen naturally as the season progresses.

5. **Game Simulations**: We then simulate the remainder of the season by iterating through each scheduled game. Player inferences are adjusted for factors such as teammate and opponent quality, deployment strategies, team dynamics, and home-ice advantage. For each game, we use the players' adjusted inferences to predict win probabilities, projected goal differential, and overtime odds by splitting the game into 30-second chunks. Scoring is modelled probabistically by chunk using a Poisson distribution. As the bootstrapped samples are linearly transformed by the adjusted inferences, this process inherently updates the probability distributions.

6. **Monte Carlo Simulations**: With game probabilities and individual player statistical distributions in hand, we run thousands of Monte Carlo simulations to conduct further probabilistic analysis. This allows us to estimate the likelihood of players reaching certain statistical milestones such as attaining 50 goals, 100 points, or winning the Art Ross trophy. Moreover, this approach enables us to project team outcomes such as playoff chances, Stanley Cup odds, or the probability of securing the first overall draft pick.

Ultimately, this process renders a model that projects future statistics and probabilities accounting for historical performance, historical peripheral statistics, age, teammate quality, teammate play styles, deployment, opponent quality, among other factors. This approach ensures that every aspect of player and team performance is modeled with both precision and flexibility, accounting for uncertainty and randomness in the game.

## Technology Stack
- **Backend**: Python
- **Database**: Supabase
- **Frontend**: ReactJS
- **Deployment**: Hosted on Firebase

## Contributing

We welcome your input and ideas to make IceAnalytics even better! If you have a feature request, suggestion for the site or the projection model, or notice an issue with the platform, we'd love to hear from you. Here's how you can contribute:

- **Submit an Issue**: Create a GitHub issue in our repository with details about your feature request, suggestion, or bug report. Be as descriptive as possible so we can understand and address it effectively.
- **Engage in Discussions**: Feel free to comment on existing issues or join discussions to help shape the future of IceAnalytics.

Your feedback is invaluable in guiding the platform's development. Thank you for helping us improve IceAnalytics!

## License

IceAnalytics is licensed under the [GPLv3.0 License](LICENSE). You are free to use, modify, and distribute the code under the terms of this license, provided all derivative works remain open-source.

## Contact

We love hearing from our users! Whether you have a question, suggestion, or just want to say hello, feel free to reach out:

- **Email**: hello@iceanalytics.ca
- **GitHub**: [IceAnalytics GitHub Repository](https://github.com/andrewderango/IceAnalytics)

Thank you for visiting IceAnalytics. We hope you enjoy exploring the insights and projections we provide!
