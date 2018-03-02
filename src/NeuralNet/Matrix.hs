module NeuralNet.Matrix (
  sumRows,
  dims,
  dimsEq,
  zero
) where

import Numeric.LinearAlgebra

zero :: Int -> Int -> Matrix Double
zero r c = (r><c) (repeat 0)

-- TODO: Check if a function available for this
sumRows :: Matrix Double -> Matrix Double
sumRows m = col (map sum (toLists m))

-- TODO: Have a look at size
dims :: Matrix a -> (Int, Int)
dims m = (rows m, cols m)

dimsEq :: Matrix a -> Matrix b -> Bool
dimsEq x y = dims x == dims y
