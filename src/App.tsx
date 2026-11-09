import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { loadPalmModel, predictRF, predictSVR } from './palm/predict.ts'
import type { PalmModel } from './palm/predict.ts'

type Prediction = {
  input: number[]
  rf: number
  svr: number
}

const integerFeatures = new Set(['Tree_Age_Years', 'Pest_Disease_Index'])
const modelComparisons = [
  { key: 'svr_thesis_settings', label: 'SVR thesis settings' },
  { key: 'svr_tuned', label: 'SVR tuned' },
  { key: 'random_forest', label: 'Random Forest' },
] as const

function App() {
  const [model, setModel] = useState<PalmModel | null>(null)
  const [values, setValues] = useState<number[]>([])
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [selectedExample, setSelectedExample] = useState<number | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [loadAttempt, setLoadAttempt] = useState(0)
  const exampleCursor = useRef(0)

  useEffect(() => {
    let active = true

    loadPalmModel()
      .then((loadedModel) => {
        const initialValues = loadedModel.features.map(
          (feature) => loadedModel.defaults[feature],
        )

        if (active) {
          setModel(loadedModel)
          setValues(initialValues)
        }
      })
      .catch(() => {
        if (active) {
          setLoadError('We couldn’t load the model. Check your connection and try again.')
        }
      })

    return () => {
      active = false
    }
  }, [loadAttempt])

  useEffect(() => {
    if (!model || values.length !== model.features.length) return

    let active = true

    Promise.all([predictRF(values), predictSVR(values)])
      .then(([rf, svr]) => {
        if (!Number.isFinite(rf) || !Number.isFinite(svr)) {
          throw new Error('The model returned an invalid prediction')
        }

        if (active) {
          setPrediction({ input: values, rf, svr })
          setPredictionError(null)
        }
      })
      .catch(() => {
        if (active) {
          setPredictionError('This prediction is unavailable. Try resetting the inputs.')
        }
      })

    return () => {
      active = false
    }
  }, [model, values])

  function changeValue(featureIndex: number, nextValue: number) {
    setValues((current) =>
      current.map((value, index) => (index === featureIndex ? nextValue : value)),
    )
    setSelectedExample(null)
    setPredictionError(null)
  }

  function resetDefaults() {
    if (!model) return

    setValues(model.features.map((feature) => model.defaults[feature]))
    setSelectedExample(null)
    setPredictionError(null)
  }

  function tryExample() {
    if (!model || model.examples.length === 0) return

    const exampleIndex = exampleCursor.current % model.examples.length
    const example = model.examples[exampleIndex]
    exampleCursor.current = (exampleIndex + 1) % model.examples.length

    setValues(model.features.map((feature) => example[feature]))
    setSelectedExample(exampleIndex)
    setPredictionError(null)
  }

  const isPredicting =
    model !== null && !predictionError && prediction?.input !== values
  const atDefaults =
    model !== null &&
    values.every((value, index) => value === model.defaults[model.features[index]])
  const activeExample =
    model && selectedExample !== null ? model.examples[selectedExample] : null
  const rfDisplay = prediction && !predictionError ? prediction.rf.toFixed(2) : '—'
  const svrDisplay = prediction && !predictionError ? prediction.svr.toFixed(2) : '—'
  const statusLabel =
    loadError || predictionError
      ? 'Unavailable'
      : !model
        ? 'Loading'
        : isPredicting
          ? 'Updating'
          : 'Live'
  const featureImportance = model
    ? Object.entries(model.feature_importance)
        .map(([feature, importance]) => ({
          feature,
          label: model.labels[model.features.indexOf(feature)] ?? feature,
          importance,
        }))
        .sort((first, second) => second.importance - first.importance)
    : []
  const largestImportance = featureImportance[0]?.importance || 1

  return (
    <div className="min-h-svh bg-canvas text-palm-950">
      <main className="mx-auto max-w-6xl px-4 pt-8 pb-32 sm:px-8 sm:pt-12 lg:pb-12">
        <header className="flex items-start gap-3 sm:gap-4">
          <img
            src="/palm/leaf.svg"
            alt=""
            width="48"
            height="48"
            className="mt-1 size-11 shrink-0 sm:size-12"
            aria-hidden="true"
          />
          <div className="max-w-2xl">
            <h1 className="text-[1.8rem] leading-tight font-semibold tracking-tight sm:text-4xl">
              Oil Palm Yield Predictor
            </h1>
            <p className="mt-3 text-sm leading-6 text-muted sm:text-base sm:leading-7">
              Thesis project, extended with a Random Forest. Runs live in your browser.
            </p>
          </div>
        </header>

        <div className="mt-8 grid items-start gap-6 lg:mt-10 lg:grid-cols-[0.85fr_1.25fr]">
          <div className="space-y-4 lg:sticky lg:top-6">
            <section
              aria-labelledby="yield-heading"
              aria-busy={isPredicting || (!model && !loadError)}
              className="relative overflow-hidden rounded-3xl bg-palm-900 p-5 text-canvas shadow-[0_12px_40px_rgba(32,75,61,0.12)] sm:p-7"
            >
              <img
                src="/palm/leaf.svg"
                alt=""
                aria-hidden="true"
                className="pointer-events-none absolute -top-9 -right-9 size-44 -rotate-12 opacity-10"
              />
              <div className="relative flex items-center justify-between gap-3">
                <h2 id="yield-heading" className="text-xs font-medium tracking-[0.16em] uppercase">
                  Predicted yield
                </h2>
                <span className="flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-xs">
                  <span aria-hidden="true" className="size-1.5 rounded-full bg-lime" />
                  {statusLabel}
                </span>
              </div>

              <div className="relative mt-7" role="status" aria-live="polite" aria-atomic="true">
                <div className="flex items-end gap-4 sm:gap-6">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-canvas/80">Random Forest</p>
                    <p className="mt-2 flex items-baseline gap-2">
                      <span className="text-[clamp(3.5rem,15vw,5rem)] leading-none font-semibold tracking-tight tabular-nums">
                        {rfDisplay}
                      </span>
                      <span className="text-sm text-canvas/70">t/ha</span>
                    </p>
                  </div>
                  <div className="shrink-0 border-l border-white/20 pb-1 pl-4">
                    <p className="text-xs text-canvas/70">Tuned SVR</p>
                    <p className="mt-2 text-2xl leading-none font-medium tabular-nums">
                      {svrDisplay}
                    </p>
                    <p className="mt-1 text-xs text-canvas/70">t/ha</p>
                  </div>
                </div>

                {activeExample && (
                  <div className="mt-6 flex items-center justify-between gap-3 border-t border-white/15 pt-4">
                    <div>
                      <p className="text-xs text-canvas/70">Actual yield</p>
                      <p className="mt-1 text-xl font-semibold tabular-nums">
                        {activeExample.actual.toFixed(2)}{' '}
                        <span className="text-xs font-normal text-canvas/70">t/ha</span>
                      </p>
                    </div>
                    <span className="rounded-full bg-white/10 px-3 py-1.5 text-xs text-canvas/80">
                      Example {(selectedExample ?? 0) + 1} / {model?.examples.length}
                    </span>
                  </div>
                )}
              </div>

              <p className="relative mt-6 text-xs leading-5 text-canvas/70">
                Two models, the same field. Adjust a slider to see how their estimates compare.
              </p>
            </section>

            {predictionError && (
              <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                {predictionError}
              </p>
            )}

            <button
              type="button"
              onClick={tryExample}
              disabled={!model || model.examples.length === 0}
              className="flex min-h-14 w-full cursor-pointer items-center justify-between gap-4 rounded-2xl bg-lime px-5 py-3 text-sm font-semibold text-palm-950 hover:bg-lime/80 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-palm-700 disabled:cursor-wait disabled:opacity-50 motion-safe:transition-colors"
            >
              Try a real example
              <span aria-hidden="true" className="text-xl">→</span>
            </button>
            <p className="px-1 text-xs leading-5 text-muted">
              Load a dataset example and compare the predictions with its actual yield.
            </p>
          </div>

          {model ? (
            <section
              aria-labelledby="conditions-heading"
              className="rounded-3xl border border-palm-900/10 bg-white p-5 shadow-[0_2px_24px_rgba(32,75,61,0.03)] sm:p-7"
            >
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-palm-900/10 pb-5">
                <div>
                  <h2 id="conditions-heading" className="text-lg font-semibold tracking-tight">
                    Field conditions
                  </h2>
                  <p className="mt-1 text-xs text-muted">{model.features.length} inputs · live estimates</p>
                </div>
                <button
                  type="button"
                  onClick={resetDefaults}
                  disabled={atDefaults && !predictionError}
                  className="flex min-h-11 cursor-pointer items-center gap-2 rounded-xl border border-palm-900/10 px-3 py-2 text-xs font-medium hover:bg-palm-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-palm-700 disabled:cursor-default disabled:opacity-40 motion-safe:transition-colors"
                >
                  <span aria-hidden="true" className="text-base">↺</span>
                  Reset defaults
                </button>
              </div>

              <div className="mt-6 grid gap-x-7 gap-y-7 sm:grid-cols-2">
                {model.features.map((feature, featureIndex) => {
                  const [minimum, maximum] = model.ranges[feature]
                  const value = values[featureIndex]
                  const discrete = integerFeatures.has(feature)
                  const decimals = discrete ? 0 : 1
                  const progress = ((value - minimum) / (maximum - minimum)) * 100
                  const sliderId = `field-${featureIndex}`

                  return (
                    <div key={feature} className="min-w-0">
                      <div className="flex min-h-8 items-center justify-between gap-3">
                        <label htmlFor={sliderId} className="text-sm leading-5 font-medium">
                          {model.labels[featureIndex]}
                        </label>
                        <span
                          aria-hidden="true"
                          className="shrink-0 rounded-lg bg-canvas px-2.5 py-1 font-mono text-sm font-medium text-palm-700 tabular-nums"
                        >
                          {value.toFixed(decimals)}
                        </span>
                      </div>
                      <input
                        id={sliderId}
                        type="range"
                        min={minimum}
                        max={maximum}
                        step={discrete ? 1 : 0.1}
                        value={value}
                        onChange={(event) => changeValue(featureIndex, Number(event.currentTarget.value))}
                        aria-describedby={`${sliderId}-bounds`}
                        className="palm-slider mt-1 h-12 w-full cursor-pointer"
                        style={{ '--range-progress': `${progress}%` } as CSSProperties}
                      />
                      <div id={`${sliderId}-bounds`} className="flex justify-between text-xs text-muted tabular-nums">
                        <span><span className="sr-only">Minimum </span>{minimum.toFixed(decimals)}</span>
                        <span><span className="sr-only">Maximum </span>{maximum.toFixed(decimals)}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          ) : (
            <section className="rounded-3xl border border-palm-900/10 bg-white p-7 sm:p-9">
              {loadError ? (
                <div role="alert">
                  <h2 className="text-lg font-semibold">The model couldn’t load</h2>
                  <p className="mt-2 text-sm leading-6 text-muted">{loadError}</p>
                  <button
                    type="button"
                    onClick={() => {
                      setLoadError(null)
                      setLoadAttempt((attempt) => attempt + 1)
                    }}
                    className="mt-5 min-h-12 cursor-pointer rounded-xl bg-palm-900 px-5 py-3 text-sm font-medium text-white hover:bg-palm-700 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-palm-700"
                  >
                    Try again
                  </button>
                </div>
              ) : (
                <div role="status" className="flex items-start gap-4">
                  <span aria-hidden="true" className="mt-1 size-5 shrink-0 rounded-full border-2 border-palm-700 border-r-transparent motion-safe:animate-spin" />
                  <div>
                    <h2 className="text-base font-semibold">Loading field controls</h2>
                    <p className="mt-2 text-sm leading-6 text-muted">
                      Downloading the model once. Predictions then run locally in your browser.
                    </p>
                  </div>
                </div>
              )}
            </section>
          )}
        </div>

        {model && (
          <section
            aria-labelledby="accuracy-heading"
            className="mt-10 rounded-3xl border border-palm-900/10 bg-white p-5 sm:p-7"
          >
            <h2 id="accuracy-heading" className="text-xl font-semibold tracking-tight sm:text-2xl">
              How accurate is this?
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
              Thesis baseline: SVR (R² 0.14). This demo improves on it with a Random Forest (R² 0.71).
            </p>

            <div className="mt-7 grid items-start gap-8 lg:grid-cols-[0.85fr_1.25fr]">
              <div className="min-w-0">
                <table className="w-full text-sm" aria-describedby="metrics-explainer">
                  <caption className="pb-3 text-left text-base font-semibold">
                    Model comparison
                  </caption>
                  <thead className="border-b border-palm-900/10 text-xs text-muted">
                    <tr>
                      <th scope="col" className="w-1/2 py-3 pr-3 text-left font-medium">Model</th>
                      <th scope="col" className="py-3 pl-3 text-right font-medium">R²</th>
                      <th scope="col" className="py-3 pl-3 text-right font-medium">RMSE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-palm-900/10">
                    {modelComparisons.map(({ key, label }) => (
                      <tr key={key} className={key === 'random_forest' ? 'bg-palm-100/70' : undefined}>
                        <th scope="row" className="py-3 pr-3 text-left font-medium">{label}</th>
                        <td className="py-3 pl-3 text-right tabular-nums">
                          {model.metrics[key].r2.toFixed(3)}
                        </td>
                        <td className="py-3 pl-3 text-right tabular-nums">
                          {model.metrics[key].rmse.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p id="metrics-explainer" className="mt-3 text-xs leading-5 text-muted">
                  Higher R² and lower RMSE mean better performance. RMSE is in t/ha.
                </p>
              </div>

              <figure aria-describedby="importance-explainer" className="min-w-0">
                <figcaption className="text-base font-semibold">
                  Random Forest feature importance
                </figcaption>
                <p id="importance-explainer" className="mt-2 text-xs leading-5 text-muted">
                  Share of model importance; bars are scaled to the largest feature.
                </p>
                <ol className="mt-5 space-y-3.5">
                  {featureImportance.map(({ feature, label, importance }) => (
                    <li key={feature}>
                      <div className="flex items-start justify-between gap-3 text-xs sm:text-sm">
                        <span className="text-muted">{label}</span>
                        <span className="shrink-0 font-medium tabular-nums">
                          {(importance * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div aria-hidden="true" className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-palm-100">
                        <div
                          className="h-full rounded-full bg-palm-700"
                          style={{ width: `${(importance / largestImportance) * 100}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ol>
              </figure>
            </div>

            <p className="mt-7 rounded-2xl bg-canvas p-4 text-sm leading-6 text-muted">
              Trained on a 5,000-row synthetic palm climate dataset. This is a portfolio demo and not for real farm decisions.
            </p>
          </section>
        )}

        <footer className="mt-8 flex flex-wrap justify-between gap-x-6 gap-y-2 border-t border-palm-900/10 pt-5 text-xs leading-5 text-muted">
          <p>t/ha = tonnes per hectare. Demo estimates, not field recommendations.</p>
          <p>Your inputs stay in your browser.</p>
        </footer>
      </main>

      {prediction && model && !predictionError && (
        <div
          aria-hidden="true"
          className="fixed inset-x-0 bottom-0 z-20 border-t border-palm-900/10 bg-canvas/95 px-5 pt-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] shadow-[0_-4px_24px_rgba(32,75,61,0.06)] backdrop-blur-md lg:hidden"
        >
          <div className="mx-auto flex max-w-2xl items-center justify-between gap-4">
            <div>
              <p className="text-[11px] font-medium text-muted">Random Forest{isPredicting ? ' · updating' : ''}</p>
              <p className="mt-0.5 text-2xl font-semibold tracking-tight tabular-nums">
                {rfDisplay} <span className="text-xs font-normal text-muted">t/ha</span>
              </p>
            </div>
            <div className="border-l border-palm-900/15 pl-5 text-right">
              <p className="text-[11px] text-muted">Tuned SVR</p>
              <p className="mt-0.5 text-lg font-medium tabular-nums">
                {svrDisplay} <span className="text-xs font-normal text-muted">t/ha</span>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
