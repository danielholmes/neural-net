module NeuralNet.Train (
  nnBackward
) where

import NeuralNet.Activation
import NeuralNet.Net
import NeuralNet.Layer
import NeuralNet.Example
import NeuralNet.Matrix
import Data.Matrix


nnBackward :: NeuralNet -> [ForwardPropStep] -> ExampleSet -> [(Matrix Double, Matrix Double)]
nnBackward nn steps examples
  | numLayers /= asSize - 1 = error "should be one entry for each layer"
  | otherwise               = result
    where
      layers = nnLayers nn
      numLayers = length layers
      as = map forwardPropA steps
      asSize = length as
      y = exampleSetY examples
      m = exampleSetM examples
      lastLayer = nnBackwardLastLayer y m (as!!numLayers) (as!!(numLayers - 1))
      (_, _, dZ) = lastLayer
      secondLastLayer = nnBackwardMidLayer (layers!!1) (layers!!0) (steps!!0) dZ
      result = map (\(dw, db, _) -> (dw, db)) [secondLastLayer, lastLayer]

nnBackwardMidLayer :: NeuronLayer -> NeuronLayer -> ForwardPropStep -> Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
nnBackwardMidLayer layer prevLayer step prevDZ = (dW, db, dZ)
  where
    a = forwardPropA step
    z = forwardPropZ step
    m = fromIntegral (ncols a)
    dZ = (transpose (layerW prevLayer) * prevDZ) * (backward (layerActivation layer) z)
    dW = mapMatrix (/ m) (dZ * transpose a)
    db = mapMatrix (/ m) dZ -- (sum dZ)

nnBackwardLastLayer :: Matrix Double -> Int -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
nnBackwardLastLayer y m asn asnm1 = (dw, db, dz)
  where
    dz = asn - y
    dw = mapMatrix (/ fromIntegral m) (dz * transpose asnm1)
    db = mapMatrix (/ fromIntegral m) dz
