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
import Numeric.LinearAlgebra

data NeuronLayer = NeuronLayer Activation (Matrix Double) (Matrix Double)
  deriving (Show, Eq)

data ForwardPropStep = ForwardPropStep (Matrix Double) (Matrix Double)
  deriving (Show, Eq)

forwardPropA :: ForwardPropStep -> Matrix Double
forwardPropA (ForwardPropStep a _) = a

forwardPropZ :: ForwardPropStep -> Matrix Double
forwardPropZ (ForwardPropStep _ z) = z

createInputForwardPropStep :: Matrix Double -> ForwardPropStep
createInputForwardPropStep a = createForwardPropStep a (zero (rows a) (cols a))

createForwardPropStep :: Matrix Double -> Matrix Double -> ForwardPropStep
createForwardPropStep a z
  | rows a /= rows z = error "a and z need same num rows"
  | cols a /= cols z = error "a and z need same num cols"
  | otherwise          = ForwardPropStep a z

layerForward :: NeuronLayer -> [Double] -> [Double]
layerForward l inputs = concat (toLists (takeColumns 1 (forwardPropA (layerForwardSet l x))))
  where x = col inputs

layerForwardSet :: NeuronLayer -> Matrix Double -> ForwardPropStep
layerForwardSet l x
  | expInputs /= n = error ("Layer " ++ show l ++ " expected " ++ show expInputs ++ " inputs but got " ++ show n)
  | otherwise      = createForwardPropStep a z
    where
      expInputs = layerNumInputs l
      n = rows x
      w = layerW l
      b = layerB l
      z = w <> x + b
      a = forward (layerActivation l) z

layerW :: NeuronLayer -> Matrix Double
layerW (NeuronLayer _ w _) = w

layerNumInputs :: NeuronLayer -> Int
layerNumInputs = cols . layerW

layerB :: NeuronLayer -> Matrix Double
layerB (NeuronLayer _ _ b) = b

layerActivation :: NeuronLayer -> Activation
layerActivation (NeuronLayer a _ _) = a

updateLayerParams :: NeuronLayer -> Matrix Double -> Matrix Double -> NeuronLayer
updateLayerParams l newW newB
  | dims (layerW l) /= dims newW = error "Ws don't align"
  | dims (layerB l) /= dims newB = error "bs don't align"
  | otherwise                    = NeuronLayer (layerActivation l) newW newB
