module NeuralNet.Layer (
  NeuronLayer (NeuronLayer),
  layerW,
  layerB,
  layerActivation,
  layerForward,
  layerNumInputs
) where

import qualified NeuralNet.Activation as Activation
import Data.Matrix
import qualified Data.Vector as Vector

data NeuronLayer = NeuronLayer Activation.Activation (Matrix Double) [Double]
  deriving (Show, Eq)

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l x
  | expInputs /= xLen = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show xLen)
  | otherwise         = Vector.toList (getCol 1 a)
    where
      expInputs = layerNumInputs l
      xLen = length x
      xMatrix = colVector (Vector.fromList x)
      wt = transpose (layerW l)
      b = colVector (Vector.fromList (layerB l))
      z = wt * xMatrix + b
      a = Activation.forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = nrows . layerW

layerB :: NeuronLayer -> [Double]
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation.Activation
layerActivation (NeuronLayer a _ _) = a
