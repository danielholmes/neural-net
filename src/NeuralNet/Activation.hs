module NeuralNet.Activation (
  Activation (Sigmoid, Tanh, ReLU),
  forward,
  backward
) where

import Data.Matrix
import NeuralNet.Matrix

data Activation = Sigmoid | Tanh | ReLU
  deriving (Eq, Show)

e :: Double
e = exp 1

forward :: Activation -> Matrix Double -> Matrix Double
forward Sigmoid = mapMatrix (\x -> 1 / (1 + (e ** (-x))))
forward Tanh = mapMatrix (\x -> ((e ** x) - (e ** (-x))) / (e ** x + e ** (-x)))
forward ReLU = mapMatrix (max 0)

backward :: Activation -> Matrix Double -> Matrix Double
backward Sigmoid = mapMatrix (\a -> a * (1 - a))
backward Tanh = mapMatrix (** 2)
backward ReLU = mapMatrix (\x -> if x < 0 then 0 else 1)
