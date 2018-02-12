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

data NeuronLayer = NeuronLayer Activation (Matrix Double) [Double]
  deriving (Show, Eq)

data ForwardPropStep = ForwardPropStep
  { forwardPropA :: Matrix Double
  , forwardPropZ :: Matrix Double
  }

createInputForwardPropStep :: Matrix Double -> ForwardPropStep
createInputForwardPropStep a = ForwardPropStep { forwardPropA = a, forwardPropZ = error "Undefined" }

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l x = Vector.toList (getCol 1 (forwardPropA (layerForwardSet l (fromList (length x) 1 x))))

layerForwardSet :: NeuronLayer -> Matrix Double -> ForwardPropStep
layerForwardSet l x
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = ForwardPropStep { forwardPropA = a, forwardPropZ = z }
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
