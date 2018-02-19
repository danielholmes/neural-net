module NeuralNet.Train (
  nnBackward,
  updateNNParams
) where

import NeuralNet.Net
import NeuralNet.Layer
import NeuralNet.Example
import NeuralNet.Activation
import NeuralNet.Matrix
import Data.Matrix


nnBackward :: NeuralNet -> [ForwardPropStep] -> ExampleSet -> [(Matrix Double, Matrix Double)]
nnBackward nn steps examples
  | numLayers /= stepsSize - 1 = error ("should be one entry (" ++ show stepsSize ++ ") for each layer (" ++ show numLayers ++ ")")
  | otherwise                  = reverse (nnBackwardStep layers orderedSteps dAL)
    where
      layers = reverse (nnLayers nn)
      orderedSteps = reverse steps
      numLayers = length layers
      stepsSize = length steps

      y = exampleSetY examples
      al = forwardPropA (head orderedSteps)
      unitRowVector = matrix 1 (ncols y) (const 1)
      dAL1 = elementwise (/) y al
      dAL2 = elementwise (/) (unitRowVector - y) (unitRowVector - al)
      dAL = -(elementwise (-) dAL1 dAL2)

nnBackwardStep :: [NeuronLayer] -> [ForwardPropStep] -> Matrix Double -> [(Matrix Double, Matrix Double)]
nnBackwardStep [] _ _ = []
nnBackwardStep _ [] _ = error "Shouldn't get here"
nnBackwardStep _ [_] _ = error "Shouldn't get here"
nnBackwardStep (l:ls) (s:ss@(ps:_)) prevDA = (dW, db) : nnBackwardStep ls ss dA
  where
    (dA, dW, db) = nnBackwardLayer l prevDA (forwardPropA ps) (forwardPropZ s)

nnBackwardLayer :: NeuronLayer -> Matrix Double -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
nnBackwardLayer l dA prevA z = (dANext, dW, db)
  where
    m = fromIntegral (ncols prevA)
    dZ = backward (layerActivation l) dA z
    -- would these be quicker as an elementwise (/) (matrix r c (const m)) ? this last matrix could be cached
    dW = scaleMatrix (1/m) (dZ * transpose prevA)
    db = scaleMatrix (1/m) (sumRows dZ)
    dANext = transpose (layerW l) * dZ

updateNNParams :: NeuralNet -> [(Matrix Double, Matrix Double)] -> Double -> NeuralNet
updateNNParams nn grads learningRate
  | length layers /= length grads = error "Must provide one grad per layer"
  | otherwise                     = updateNNLayers nn newLayers
    where
      layers = nnLayers nn
      layerSets = zip layers grads
      newLayers = map (\(l, (dW, dB)) -> updateLayerParamsFromGrads l dW dB learningRate) layerSets

updateLayerParamsFromGrads :: NeuronLayer -> Matrix Double -> Matrix Double -> Double -> NeuronLayer
updateLayerParamsFromGrads l dW db learningRate = updateLayerParams l newDw newDb
  where
    newDw = elementwise (-) (layerW l) (scaleMatrix learningRate dW)
    newDb = elementwise (-) (layerB l) (scaleMatrix learningRate db)
