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

1. **Data Collection**: We scrape player and team statistical and biographical data from the NHL API for every season since 2007, widely considered to be the start of the analytics era in the NHL. The 2006-07 NHL season was the first to have publicized shot data became available and enable advanced models like expected goals (xG). These advanced analytics data provide a rich foundation for the modeling done for this engine. Scraping is done for the concurrent season's data as well, including player statistics, team performance metrics, and the complete game schedule to enable rest-of-season projections.

2. **Model Training**: Leveraging the scrpaed historical data, our engine trains a series of models to infer metrics such as games played (GP), average time on ice (ATOI), goals (G), primary assists (A1), secondary assits (A2), and many defensive analytics such as team-level goals against (GA). These models include support vector regression, neural networks, and XGBoost, whose hyperparameters were optimized via Optuna.

3. **Inferencing**: Once the models are trained, we make inferences for the current season by integrating the predictions from these models with each player's ongoing performance. Subsequently, a Savitzky-Golay filter is applied for calibrative scaling. This helps the inferences better align with historical scaling and ensure a more accurate trend in player and team performance from season to season.

4. **Bootstrapping and PRUS**: To account for variability and uncertainty, we train 500 bootstrapped models for each key statistic: ATOI, G, A1, A2 and GP. Prior to the bootstrap sampling, the data is split into a test and train dataset, enabling us to compute the variance of the bootstrap models' residuals in parallel. This idea fundamentally underpins a novel algorithm that we are calling the Parallelized Residual-based Uncertainty Scaling (PRUS) algorithm, which is critical in accounting for epistemic uncertainty. If we scale the variance of the predictions across the 500 bootstraps for a specific player by the computed residual-based variances, we can estimate the total uncertainty of the player's performance and effectively generate a probability distribution for each of the key target variables. This approach captures both *epistemic* and *aleatoric* uncertainty, while dynamically adjusting based on the player's current-season performance and its progression.

5. **Game Simulations**: We then simulate the remainder of the season by looping through each scheduled game. Player inferences are adjusted for factors such as teammate and opponent quality, deployment strategies, team dynamics, and home-ice advantage. For each game, we use the players' adjusted inferences to predict win probabilities, projected goal differential, and overtime odds by splitting the game into 30-second chunks and using a Poisson distribution for scoring. As the bootstrapped samples mask the adjusted inferences, this process inherently updates the probability distributions.

6. **Monte Carlo Simulations**: With game probabilities and individual player statistical distributions in hand, we run thousands of Monte Carlo simulations to conduct further probabilistic analysis. This allows us to estimate the likelihood of players reaching certain statistical milestones such as attaining 50 goals, 100 points, or winning the Art Ross trophy. Moreover, this approach enables us to project team outcomes such as playoff chances, Stanley Cup odds, or the probability of securing the first overall draft pick.

Ultimately, this process renders a model that projects future statistics and probabilities accounting for historical performance, historical peripheral statistics, age, teammate quality, teammate play styles, deployment, opponent quality, among other factors. This approach ensures that every aspect of player and team performance is modeled with both precision and flexibility, accounting for uncertainty and randomness in the game.

## Technology Stack
- **Backend**: Python, using libraries like SciPy, scikit-learn, XGBoost, and TensorFlow.
- **Database**: Supabase for data storage and management.
- **Frontend**: ReactJS, styled with JSX and SCSS.
- **Deployment**: Hosted on Firebase for high availability.

## Contributing

We welcome contributions! Whether it's improving the projection model, fixing bugs, or adding new features, feel free to submit pull requests. Here's how you can get started:

1. Fork the repository.
2. Create a new branch for your feature: 
```
git checkout -b feature-name
```
3. Make changes and commit them: 
```
git commit -m "Add new feature".
```
4. Push your changes: 
```
git push origin feature-name
```
5. Submit a pull request.


## License

IceAnalytics is licensed under the [GPLv3.0 License](LICENSE). You are free to use, modify, and distribute the code under the terms of this license, provided all derivative works remain open-source.

## Contact

We love hearing from our users! Whether you have a question, suggestion, or just want to say hello, feel free to reach out:

- **Email**: contact@iceanalytics.ca
- **GitHub**: [IceAnalytics GitHub Repository](https://github.com/andrewderango/IceAnalytics)

Thank you for visiting IceAnalytics. We hope you enjoy exploring the insights and projections we provide!
