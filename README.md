# Oil Palm Yield Predictor

A mobile-first demo for estimating oil palm yield from 10 climate and field inputs.  
Extends an SVR thesis baseline with a Random Forest; all predictions run in your browser.

## Results

| Model | R² | RMSE (t/ha) |
| --- | ---: | ---: |
| SVR thesis settings | 0.14 | 0.59 |
| Tuned SVR | 0.35 | 0.52 |
| Random Forest | 0.71 | 0.34 |

Higher R² and lower RMSE indicate better performance. RMSE is in tonnes per hectare.

## How it works

The models were trained in Python with scikit-learn and exported to JSON. The app fetches and caches `public/palm/palm-model.json`, then uses TypeScript to run the tuned SVR and Random Forest entirely in the browser as you adjust the sliders.

## Synthetic data

Trained on a 5,000-row synthetic palm climate dataset. This is a portfolio demo and not for real farm decisions.

## Tech stack

Vite, React, TypeScript, and Tailwind CSS v4.

**Live demo:** https://palm-yield-predictor.vercel.app
