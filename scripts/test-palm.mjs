import { readFile } from 'node:fs/promises'
import { predictRF, predictSVR } from '../src/palm/predict.ts'

const modelJSON = await readFile(
  new URL('../public/palm/palm-model.json', import.meta.url),
  'utf8',
)
const model = JSON.parse(modelJSON)

globalThis.fetch = async (url) => {
  if (url !== '/palm/palm-model.json') {
    throw new Error(`Unexpected model URL: ${url}`)
  }

  return new Response(modelJSON)
}

function printRow(values) {
  console.log(values.map((value) => String(value).padEnd(12)).join('').trimEnd())
}

printRow(['Example', 'Actual', 'RF', 'SVR'])

for (const [exampleIndex, example] of model.examples.entries()) {
  const input = model.features.map((feature) => example[feature])
  const [rf, svr] = await Promise.all([predictRF(input), predictSVR(input)])

  printRow([
    exampleIndex + 1,
    example.actual.toFixed(4),
    rf.toFixed(4),
    svr.toFixed(4),
  ])
}
