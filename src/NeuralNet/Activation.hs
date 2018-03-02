module NeuralNet.Activation (
  Activation (Sigmoid, Tanh, ReLU),
  forward,
  backward
) where

import Numeric.LinearAlgebra

data Activation = Sigmoid | Tanh | ReLU
  deriving (Eq, Show)

e :: Double
e = exp 1

forward :: Activation -> Matrix Double -> Matrix Double
forward Sigmoid = cmap (\x -> 1 / (1 + (e ** (-x))))
forward Tanh = cmap (\x -> ((e ** x) - (e ** (-x))) / (e ** x + e ** (-x)))
forward ReLU = cmap (max 0)

backward :: Activation -> Matrix Double -> Matrix Double -> Matrix Double
backward Sigmoid dA z = dA * (s * (1 - s))
  where s = cmap (\x -> 1 / (1 + exp x)) (-z)
backward Tanh _ _ = error "Unsure of impl"
backward ReLU dA z = dA * step z
