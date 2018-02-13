module NeuralNet.Train (
  nnBackward
) where

import NeuralNet.Activation
import NeuralNet.Net
import NeuralNet.Layer
import NeuralNet.Example
import NeuralNet.Matrix
import Data.Matrix
import Debug.Trace


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
    -- np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dZFirst = traceShow ("a", a, "z", z , "w", layerW prevLayer, "prevDZ", prevDZ) (transpose (layerW prevLayer) * prevDZ)
    dZSecond = backward (layerActivation layer) z
    dZ = traceShow ("dZ1 calcs", dZFirst) (dZFirst * dZSecond)
    dW = traceShow ("dW calcs", "dZ1", dZ, "X", a) (mapMatrix (/ m) (dZ * transpose a))
    db = mapMatrix (/ m) (sumRows dZ)

nnBackwardLastLayer :: Matrix Double -> Int -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, Matrix Double)
nnBackwardLastLayer y m asn asnm1 = (dW, db, dZ)
  where
    dZ = asn - y
    dW = mapMatrix (/ fromIntegral m) (dZ * transpose asnm1)
    db = mapMatrix (/ fromIntegral m) (sumRows dZ)
