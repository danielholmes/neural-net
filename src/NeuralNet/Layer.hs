module NeuralNet.Layer (
  NeuronLayer (NeuronLayer),
  layerW,
  layerB,
  layerActivation,
  layerNumInputs,
  layerForward,
  layerForwardSet
) where

import NeuralNet.Activation
import NeuralNet.Example
import Data.Matrix
import qualified Data.Vector as Vector

data NeuronLayer = NeuronLayer Activation (Matrix Double) [Double]
  deriving (Show, Eq)

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l x = Vector.toList (getCol 1 rawResult)
  where
    examples = createExampleSet [(x, error "Undefined")]
    rawResult = layerForwardSet l examples


layerForwardSet :: NeuronLayer -> ExampleSet -> Matrix Double
layerForwardSet l examples
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = a
    where
      expInputs = layerNumInputs l
      n = exampleSetN examples
      x = exampleSetX examples
      wt = transpose (layerW l)
      b = colVector (Vector.fromList (layerB l))
      z = wt * x + b
      a = forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = nrows . layerW

layerB :: NeuronLayer -> [Double]
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
