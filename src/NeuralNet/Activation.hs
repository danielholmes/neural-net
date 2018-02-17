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

backward :: Activation -> Matrix Double -> Matrix Double -> Matrix Double
backward Sigmoid dA z = elementwise (*) dA (elementwise (*) s (mapMatrix (1-) s)) -- mapMatrix (\a -> a * (1 - a))
  where s = mapMatrix (\x -> 1 / (1 + exp x)) (-z)
backward Tanh _ _ = error "Unsure of impl"
backward ReLU dA z = elementwise (*) dA scale
  where scale = mapMatrix (\x -> if x <= 0 then 0 else 1) z
