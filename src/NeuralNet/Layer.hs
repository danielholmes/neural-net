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
import NeuralNet.Matrix
import Data.Matrix
import qualified Data.Vector as Vector

data NeuronLayer = NeuronLayer Activation (Matrix Double) [Double]
  deriving (Show, Eq)

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l x = Vector.toList (getCol 1 (layerForwardSet l (fromList (length x) 1 x)))

layerForwardSet :: NeuronLayer -> Matrix Double -> Matrix Double
layerForwardSet l x
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = a
    where
      expInputs = layerNumInputs l
      n = nrows x
      m = ncols x
      wt = transpose (layerW l)
      b = colVector (Vector.fromList (layerB l))
      bProjected = projectMatrixX b m
      z = wt * x + bProjected
      a = forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = nrows . layerW

layerB :: NeuronLayer -> [Double]
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
