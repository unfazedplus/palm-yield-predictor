export type PalmModel = {
  features: string[]
  labels: string[]
  ranges: Record<string, [number, number]>
  defaults: Record<string, number>
  examples: Array<Record<string, number> & { actual: number }>
  metrics: Record<
    'svr_thesis_settings' | 'svr_tuned' | 'random_forest',
    {
      r2: number
      rmse: number
    }
  >
  feature_importance: Record<string, number>
  svr: {
    gamma: number
    intercept: number
    scaler_min: number[]
    scaler_range: number[]
    sv: number[][]
    coef: number[]
  }
  rf: {
    trees: {
      f: number[]
      t: number[]
      l: number[]
      r: number[]
      v: number[]
    }[]
  }
}

let modelPromise: Promise<PalmModel> | undefined

export function loadPalmModel(): Promise<PalmModel> {
  if (!modelPromise) {
    modelPromise = fetch('/palm/palm-model.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load palm model: ${response.status}`)
        }

        return response.json() as Promise<PalmModel>
      })
      .catch((error: unknown) => {
        modelPromise = undefined
        throw error
      })
  }

  return modelPromise
}

export async function predictSVR(x: number[]): Promise<number> {
  const { svr } = await loadPalmModel()
  const scaled = x.map(
    (value, featureIndex) =>
      (value - svr.scaler_min[featureIndex]) / svr.scaler_range[featureIndex],
  )
  let prediction = svr.intercept

  for (let supportIndex = 0; supportIndex < svr.sv.length; supportIndex++) {
    const supportVector = svr.sv[supportIndex]
    let squaredDistance = 0

    for (let featureIndex = 0; featureIndex < scaled.length; featureIndex++) {
      const difference = scaled[featureIndex] - supportVector[featureIndex]
      squaredDistance += difference * difference
    }

    prediction +=
      svr.coef[supportIndex] * Math.exp(-svr.gamma * squaredDistance)
  }

  return Math.max(1.0, prediction)
}

export async function predictRF(x: number[]): Promise<number> {
  const { rf } = await loadPalmModel()
  let total = 0

  for (const tree of rf.trees) {
    let nodeIndex = 0

    while (tree.l[nodeIndex] !== -1) {
      nodeIndex =
        x[tree.f[nodeIndex]] <= tree.t[nodeIndex]
          ? tree.l[nodeIndex]
          : tree.r[nodeIndex]
    }

    total += tree.v[nodeIndex]
  }

  return Math.max(1.0, total / rf.trees.length)
}
