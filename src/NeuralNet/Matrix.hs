module NeuralNet.Matrix (
  mapMatrix,
  broadcastCols,
  sumRows,
  dims,
  dimsEq
) where

import Data.Matrix

mapMatrix :: (a -> a) -> Matrix a -> Matrix a
mapMatrix f m = foldr (mapRow (\_ x -> f x)) m [1..(nrows m)]

broadcastCols :: Matrix a -> Int -> Matrix a
broadcastCols m n = fromLists (map (replicate n . head) (toLists m))

sumRows :: Num a => Matrix a -> Matrix a
sumRows m = fromList (nrows m) 1 (map sum (toLists m))

dims :: Matrix a -> (Int, Int)
dims m = (nrows m, ncols m)

dimsEq :: Matrix a -> Matrix b -> Bool
dimsEq x y = dims x == dims y
