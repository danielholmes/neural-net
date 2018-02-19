module NeuralNet.Layer (
  NeuronLayer (NeuronLayer),
  ForwardPropStep (),
  forwardPropA,
  forwardPropZ,
  createInputForwardPropStep,
  createForwardPropStep,
  layerW,
  layerB,
  layerActivation,
  layerNumInputs,
  layerForward,
  layerForwardSet,
  updateLayerParams
) where

import NeuralNet.Activation
import NeuralNet.Matrix
import Data.Matrix
import qualified Data.Vector as Vector

data NeuronLayer = NeuronLayer Activation (Matrix Double) (Matrix Double)
  deriving (Show, Eq)

data ForwardPropStep = ForwardPropStep (Matrix Double) (Matrix Double)

forwardPropA :: ForwardPropStep -> Matrix Double
forwardPropA (ForwardPropStep a _) = a

forwardPropZ :: ForwardPropStep -> Matrix Double
forwardPropZ (ForwardPropStep _ z) = z

createInputForwardPropStep :: Matrix Double -> ForwardPropStep
createInputForwardPropStep a = createForwardPropStep a (zero (nrows a) (ncols a))

createForwardPropStep :: Matrix Double -> Matrix Double -> ForwardPropStep
createForwardPropStep a z
  | nrows a /= nrows z = error "a and z need same num rows"
  | ncols a /= ncols z = error "a and z need same num cols"
  | otherwise          = ForwardPropStep a z

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l inputs = Vector.toList (getCol 1 (forwardPropA (layerForwardSet l x)))
  where x = colVector (Vector.fromList inputs)

layerForwardSet :: NeuronLayer -> Matrix Double -> ForwardPropStep
layerForwardSet l x
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = createForwardPropStep a z
    where
      expInputs = layerNumInputs l
      n = nrows x
      m = ncols x
      w = layerW l
      b = layerB l
      bBroadcasted = broadcastCols b m
      z = w * x + bBroadcasted
      a = forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = ncols . layerW

layerB :: NeuronLayer -> Matrix Double
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a

updateLayerParams :: NeuronLayer -> Matrix Double -> Matrix Double -> NeuronLayer
updateLayerParams l newW newB
  | dims (layerW l) /= dims newW = error "Ws don't align"
  | dims (layerB l) /= dims newB = error "bs don't align"
  | otherwise                    = NeuronLayer (layerActivation l) newW newB
