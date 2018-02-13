module NeuralNet.Layer (
  NeuronLayer (NeuronLayer),
  ForwardPropStep (),
  forwardPropA,
  forwardPropZ,
  createInputForwardPropStep,
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

data NeuronLayer = NeuronLayer Activation (Matrix Double) (Matrix Double)
  deriving (Show, Eq)

data ForwardPropStep = ForwardPropStep
  { forwardPropA :: Matrix Double
  , forwardPropZ :: Matrix Double
  }

createInputForwardPropStep :: Matrix Double -> ForwardPropStep
createInputForwardPropStep a = ForwardPropStep { forwardPropA = a, forwardPropZ = zero 0 0 }

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l inputs = Vector.toList (getCol 1 (forwardPropA (layerForwardSet l x)))
  where x = colVector (Vector.fromList inputs)

layerForwardSet :: NeuronLayer -> Matrix Double -> ForwardPropStep
layerForwardSet l x
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = ForwardPropStep { forwardPropA = a, forwardPropZ = z }
    where
      expInputs = layerNumInputs l
      n = nrows x
      m = ncols x
      w = layerW l
      b = layerB l
      bProjected = broadcastCols b m
      z = w * x + bProjected
      a = forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = ncols . layerW

layerB :: NeuronLayer -> Matrix Double
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a
