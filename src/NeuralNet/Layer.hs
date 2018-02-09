module NeuralNet.Layer (
  NeuronLayer (NeuronLayer),
  layerW,
  layerB,
  layerActivation,
  layerForward,
  numLayerInputs
) where

import qualified NeuralNet.Activation as Activation
import Data.Matrix
import qualified Data.Vector as Vector

data NeuronLayer = NeuronLayer Activation.Activation (Matrix Double) [Double]
  deriving (Show, Eq)

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l x = Vector.toList (getCol 1 a)
  where
    xMatrix = colVector (Vector.fromList x)
    wt = transpose (layerW l)
    b = colVector (Vector.fromList (layerB l))
    z = wt * xMatrix + b
    a = Activation.forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

numLayerInputs :: NeuronLayer -> Int
numLayerInputs = nrows . layerW

layerB :: NeuronLayer -> [Double]
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation.Activation
layerActivation (NeuronLayer a _ _) = a
