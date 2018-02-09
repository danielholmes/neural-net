module NeuralNet.Activation (
  Activation (Sigmoid, Tanh, ReLU),
  forward
) where

import Data.Matrix
import NeuralNet.Matrix

data Activation = Sigmoid | Tanh | ReLU
  deriving (Eq, Show)

--e :: Double
--e = exp 1

forward :: Activation -> Matrix Double -> Matrix Double
forward Sigmoid _ = error "Hello" --  1 / (1 + (e ** (-x)))
forward Tanh _ = error "TODO"
forward ReLU m = mapMatrix (max 0) m
