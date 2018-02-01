module NeuralNet.Activation (
  Activation (Sigmoid, Tanh, ReLU),
  forward
) where

data Activation = Sigmoid | Tanh | ReLU
  deriving (Eq, Show)

e :: Double
e = exp 1

forward :: Activation -> Double -> Double
forward Sigmoid x = 1 / (1 + (e ** (-x)))
forward Tanh _ = error "TODO"
forward ReLU _ = error "TODO"
